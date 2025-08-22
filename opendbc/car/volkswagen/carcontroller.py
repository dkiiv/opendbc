import numpy as np
import math
from opendbc.can.packer import CANPacker
from opendbc.car import ACCELERATION_DUE_TO_GRAVITY, Bus, DT_CTRL, AngleSteeringLimits, structs, rate_limit
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.interfaces import CarControllerBase, ISO_LATERAL_ACCEL
from opendbc.car.volkswagen import mqbcan, pqcan
from opendbc.car.volkswagen.values import CANBUS, CarControllerParams, VolkswagenFlags
from opendbc.car.vehicle_model import VehicleModel

VisualAlert = structs.CarControl.HUDControl.VisualAlert
LongCtrlState = structs.CarControl.Actuators.LongControlState

# limit angle rate to both prevent a fault and for low speed comfort (~12 mph rate down to 0 mph)
MAX_ANGLE_RATE = 5  # deg/20ms frame,

# Add extra tolerance for average banked road since safety doesn't have the roll
AVERAGE_ROAD_ROLL = 0.06  # ~3.4 degrees, 6% superelevation. higher actual roll lowers lateral acceleration
MAX_LATERAL_ACCEL = ISO_LATERAL_ACCEL + (ACCELERATION_DUE_TO_GRAVITY * AVERAGE_ROAD_ROLL)  # ~3.6 m/s^2
MAX_LATERAL_JERK = 3.0 + (ACCELERATION_DUE_TO_GRAVITY * AVERAGE_ROAD_ROLL)  # ~3.6 m/s^3


def get_max_angle_delta(v_ego_raw: float, VM: VehicleModel):
  max_curvature_rate_sec = MAX_LATERAL_JERK / (v_ego_raw ** 2)  # (1/m)/s
  max_angle_rate_sec = math.degrees(VM.get_steer_from_curvature(max_curvature_rate_sec, v_ego_raw, 0))  # deg/s
  return max_angle_rate_sec * (DT_CTRL * CarControllerParams.STEER_STEP)


def get_max_angle(v_ego_raw: float, VM: VehicleModel):
  max_curvature = MAX_LATERAL_ACCEL / (v_ego_raw ** 2)  # 1/m
  return math.degrees(VM.get_steer_from_curvature(max_curvature, v_ego_raw, 0))  # deg


def apply_vwpla_steer_angle_limits(apply_angle: float, apply_angle_last: float, v_ego_raw: float, steering_angle: float,
                                   lat_active: bool, limits: AngleSteeringLimits, VM: VehicleModel) -> float:
  v_ego_raw = max(v_ego_raw, 1)

  # *** max lateral jerk limit ***
  max_angle_delta = get_max_angle_delta(v_ego_raw, VM)

  # prevent fault
  max_angle_delta = min(max_angle_delta, MAX_ANGLE_RATE)
  new_apply_angle = rate_limit(apply_angle, apply_angle_last, -max_angle_delta, max_angle_delta)

  # *** max lateral accel limit ***
  max_angle = get_max_angle(v_ego_raw, VM)
  new_apply_angle = np.clip(new_apply_angle, -max_angle, max_angle)

  # angle is current angle when inactive
  if not lat_active:
    new_apply_angle = steering_angle

  # prevent fault
  return float(np.clip(new_apply_angle, -limits.STEER_ANGLE_MAX, limits.STEER_ANGLE_MAX))

def get_safety_CP():
  # We use the VOLKSWAGEN_JETTA_MK6 platform for lateral limiting to match safety
  from opendbc.car.tesla.interface import CarInterface
  return CarInterface.get_non_essential_params("VOLKSWAGEN_JETTA_MK6")

def limit_jerk(accel, prev_accel, max_jerk, dt):
  max_delta_accel = max_jerk * dt
  delta_accel = max(-max_delta_accel, min(accel - prev_accel, max_delta_accel))
  return prev_accel + delta_accel

def EPB_handler(CS, self, ACS_Sta_ADR, ACS_Sollbeschl, vEgo, stopping):
  if (ACS_Sta_ADR == 1 and ACS_Sollbeschl < 0) and \
    ((CS.MOB_Standby and vEgo <= (18 * CV.KPH_TO_MS)) or self.EPB_enable):
      if not self.EPB_enable:  # First frame of EPB entry
          self.EPB_counter = 0
          self.EPB_brake = 0
          self.EPB_enable = 1
          self.EPB_enable_history = [True] * len(self.EPB_enable_history)
          self.EPB_brake_last = ACS_Sollbeschl
      else:
          self.EPB_brake = limit_jerk(-4, self.EPB_brake_last, 0.7, 0.02) if stopping else ACS_Sollbeschl
          self.EPB_brake_last = self.EPB_brake
      self.EPB_counter += 1
  else:
      if self.EPB_enable and self.EPB_counter < 10:  # Keep EPB_enable active for 10 frames
          self.EPB_counter += 1
      else:
          self.EPB_brake = 0
          self.EPB_enable = 0

  if CS.out.gasPressed or CS.out.brakePressed or CS.gra_stock_values["GRA_Abbrechen"]:
    if self.EPB_enable:
      self.ACC_anz_blind = 1
    self.EPB_brake = 0
    self.EPB_enable = 0
    self.EPB_enable_history = [False] * len(self.EPB_enable_history)

  if self.ACC_anz_blind and self.ACC_anz_blind_counter < 150:
    self.ACC_anz_blind_counter += 1
  else:
    self.ACC_anz_blind = 0
    self.ACC_anz_blind_counter = 0

  # Update EPB historical states and calculate EPB_active
  self.EPB_active = int((self.EPB_enable_history[(len(self.EPB_enable_history) - 2)] and not self.EPB_enable) or self.EPB_enable)
  self.EPB_enable_history = self.EPB_enable_history[1:] + [self.EPB_enable]

  return self.EPB_enable, self.EPB_brake, self.EPB_active

class CarController(CarControllerBase):
  def __init__(self, dbc_names, CP, CP_SP):
    super().__init__(dbc_names, CP, CP_SP)
    self.CCP = CarControllerParams(CP)
    self.CCS = pqcan if CP.flags & VolkswagenFlags.PQ else mqbcan
    self.packer_pt = CANPacker(dbc_names[Bus.pt])
    self.ext_bus = CANBUS.pt if CP.networkLocation == structs.CarParams.NetworkLocation.fwdCamera else CANBUS.cam
    self.aeb_available = not CP.flags & VolkswagenFlags.PQ

    self.apply_angle_last = 0
    self.gra_acc_counter_last = None
    self.bremse8_counter_last = None
    self.bremse11_counter_last = None
    self.acc_sys_counter_last = None
    self.acc_anz_counter_last = None
    self.ACC_anz_blind = 0
    self.ACC_anz_blind_counter = 0
    self.PLA_status = 0
    self.PLA_entryCounter = 0
    self.PLA_driverExit = False
    self.PLA_driverExit_last = False
    self.CSLH3_SignLast = 0
    self.accel_last = 0
    self.frame = 0
    self.EPB_brake = 0
    self.EPB_brake_last = 0
    self.EPB_enable = 0
    self.EPB_enable_history = [False] * 25  # 0.5s history
    self.EPB_active = 0
    self.EPB_counter = 0
    self.accel_diff = 0
    self.long_deviation = 0
    self.long_jerklimit = 0
    self.stopped = 0
    self.stopping = 0

  # Vehicle model used for lateral limiting
    self.VM = VehicleModel(get_safety_CP())

  def update(self, CC, CC_SP, CS, now_nanos):
    actuators = CC.actuators
    hud_control = CC.hudControl
    can_sends = []

    # **** Steering Controls ************************************************ #

    if CS.LH2_Abbr == 2 and CS.out.cruiseState.available:
      self.PLA_driverExit = True
    else:
      self.PLA_driverExit = False

    if self.frame % self.CCP.STEER_STEP == 0:
      # PLA_status definitions:
      #  10 = reset EPS driver torque override flag
      #  15 = standby
      #  13 = active
      #  11 = activatable, entry request signal. 11 frames required
      if CC.latActive and not self.PLA_driverExit:
        self.PLA_status = 13 if self.PLA_entryCounter >= 11 else 11
        self.PLA_entryCounter += 1 if self.PLA_entryCounter <= 32 else self.PLA_entryCounter
        # retry entry until engagement. TODO: add a counter to disable if this takes too long? (error)
        if CS.LH2_steeringState != 64 and self.PLA_entryCounter >= 30:
          self.PLA_entryCounter = 0
      else:
        self.PLA_status = 10 if self.PLA_driverExit_last and not self.PLA_driverExit else 15  # pulse reset on falling edge
        self.PLA_entryCounter = 0
        self.PLA_driverExit_last = self.PLA_driverExit

      apply_angle = apply_vwpla_steer_angle_limits(actuators.steeringAngleDeg, self.apply_angle_last, CS.out.vEgoRaw,
                                                             CS.out.steeringAngleDeg, CC.latActive,
                                                             CarControllerParams.ANGLE_LIMITS, self.VM) if self.PLA_status == 13 else CS.out.steeringAngleDeg

      self.apply_angle_last = apply_angle
      can_sends.append(self.CCS.create_steering_control(self.packer_pt, CANBUS.pt, apply_angle, self.PLA_status, self.CSLH3_SignLast))
      self.CSLH3_SignLast = CS.LH_3_Sign

      if self.CP.flags & VolkswagenFlags.STOCK_HCA_PRESENT:
        # Pacify VW Emergency Assist driver inactivity detection by changing its view of driver steering input torque
        # to the greatest of actual driver input or 2x openpilot's output (1x openpilot output is not enough to
        # consistently reset inactivity detection on straight level roads). See commaai/openpilot#23274 for background.
        ea_simulated_torque = float(np.clip(apply_torque * 2, -self.CCP.STEER_MAX, self.CCP.STEER_MAX))
        if abs(CS.out.steeringTorque) > abs(ea_simulated_torque):
          ea_simulated_torque = CS.out.steeringTorque
        can_sends.append(self.CCS.create_eps_update(self.packer_pt, CANBUS.cam, CS.eps_stock_values, ea_simulated_torque))

    # **** Acceleration Controls ******************************************** #

    if self.frame % self.CCP.ACC_CONTROL_STEP == 0 and self.CP.openpilotLongitudinalControl:
      acc_control = self.CCS.acc_control_value(CS.out.cruiseState.available, CS.out.accFaulted, CC.longActive, CC.cruiseControl.override)
      stopping = actuators.longControlState == LongCtrlState.stopping
      starting = actuators.longControlState == LongCtrlState.pid and (CS.esp_hold_confirmation or CS.out.vEgo < self.CP.vEgoStopping)
      accel = np.clip(actuators.accel, self.CCP.ACCEL_MIN, self.CCP.ACCEL_MAX) if CC.longActive else 0
                                                                            # SMA to EMA conversion: alpha = 2 / (n + 1)    n = SMA-sample
      self.accel_diff = (0.0019 * (accel - self.accel_last)) + (1 - 0.0019) * self.accel_diff         # 1000 SMA equivalence
      self.long_jerklimit = (0.01 * (clip(abs(accel), 0.7, 2))) + (1 - 0.01) * self.long_jerklimit    # set jerk limit based on accel
      self.long_deviation = np.clip(CS.out.vEgo/40, 0, 0.13) * np.interp(abs(accel - self.accel_diff), [0, .2, 1.], [0.0, 0.0, 0.0])

      EPB_handler(CS, self, acc_control, accel, CS.out.vEgoRaw, self.stopping)

        # Keep ACC status 0 while EPB "active"
      if self.EPB_active:
        acc_control = 0

      self.accel_last = accel
      if self.CCS == pqcan:
        can_sends.append(self.CCS.create_epb_control(self.packer_pt, CANBUS.body, self.EPB_brake, self.EPB_enable))
      can_sends.extend(self.CCS.create_acc_accel_control(self.packer_pt, CANBUS.pt, CS.acc_type, accel,
                                                          acc_control, stopping, starting, CS.esp_hold_confirmation,
                                                          self.long_deviation, self.long_jerklimit))

      #if self.aeb_available:
      #  if self.frame % self.CCP.AEB_CONTROL_STEP == 0:
      #    can_sends.append(self.CCS.create_aeb_control(self.packer_pt, False, False, 0.0))
      #  if self.frame % self.CCP.AEB_HUD_STEP == 0:
      #    can_sends.append(self.CCS.create_aeb_hud(self.packer_pt, False, False))

    # **** HUD Controls ***************************************************** #

    if self.frame % self.CCP.LDW_STEP == 0:
      hud_alert = 0
      if hud_control.visualAlert in (VisualAlert.steerRequired, VisualAlert.ldw):
        hud_alert = self.CCP.LDW_MESSAGES["laneAssistTakeOver"]
      if CC.latActive and CS.LH2_steeringState != 64 and self.frame % 2:
        pulse = 0
      else:
        if CC.latActive and CS.LH2_steeringState == 64:
          pulse = 0
        else:
          pulse = 1
      can_sends.append(self.CCS.create_lka_hud_control(self.packer_pt, CANBUS.pt, CS.ldw_stock_values, (CC.latActive and CS.LH2_steeringState == 64),
                                                       CS.out.steeringPressed, hud_alert, hud_control, pulse))

    if self.frame % self.CCP.ACC_HUD_STEP == 0 and self.CP.openpilotLongitudinalControl:
      lead_distance = 0
      if hud_control.leadVisible and self.frame * DT_CTRL > 1.0:  # Don't display lead until we know the scaling factor
        lead_distance = 512 if CS.upscale_lead_car_signal else 8
      acc_hud_status = self.CCS.acc_hud_status_value(CS.out.cruiseState.available, CS.out.accFaulted, CC.longActive)
      # FIXME: follow the recent displayed-speed updates, also use mph_kmh toggle to fix display rounding problem?
      set_speed = hud_control.setSpeed * CV.MS_TO_KPH
      can_sends.append(self.CCS.create_acc_hud_control(self.packer_pt, CANBUS.pt, acc_hud_status, set_speed,
                                                       lead_distance, hud_control.leadDistanceBars))

    # **** Stock ACC Button Controls **************************************** #

    if self.CP.openpilotLongitudinalControl:
      if CS.gra_stock_values["COUNTER"] != self.gra_acc_counter_last:
        can_sends.append(self.CCS.create_acc_buttons_control(self.packer_pt, self.ext_bus, CS.gra_stock_values, self.CP.openpilotLongitudinalControl,
                                                            cancel=CC.cruiseControl.cancel, resume=CC.cruiseControl.resume))
      if not (CC.cruiseControl.cancel or CC.cruiseControl.resume) and CS.out.cruiseState.enabled:
        if not self.CP.pcmCruiseSpeed:
          self.cruise_button = self.get_cruise_buttons(CS, CC.vCruise)
          if self.cruise_button is not None:
            if self.acc_type == -1:
              if self.button_count >= 2 and self.v_set_dis_prev != self.v_set_dis:
                self.acc_type = 1 if abs(self.v_set_dis - self.v_set_dis_prev) >= 10 and self.last_cruise_button in (1, 2) else \
                                0 if abs(self.v_set_dis - self.v_set_dis_prev) < 10 and self.last_cruise_button not in (1, 2) else 1
              if self.send_count >= 10 and self.v_set_dis_prev == self.v_set_dis:
                self.cruise_button = 3 if self.cruise_button == 1 else 4
            if self.acc_type == 0:
              self.cruise_button = 1 if self.cruise_button == 1 else 2  # accel, decel
            elif self.acc_type == 1:
              self.cruise_button = 3 if self.cruise_button == 1 else 4  # resume, set
            if self.frame % self.CCP.BTN_STEP == 0:
              can_sends.append(self.CCS.create_acc_buttons_control(self.packer_pt, CS.gra_stock_values, self.CP.openpilotLongitudinalControl,
                                                                  frame=(self.frame // self.CCP.BTN_STEP), buttons=self.cruise_button,
                                                                  custom_stock_long=True))
              self.send_count += 1
          else:
            self.send_count = 0
          self.last_cruise_button = self.cruise_button

    # *** Below here is for OEM+ behavior modification of OEM ACC *** #
    # Modify Motor_2, Bremse_8, Bremse_11
    if VolkswagenFlags.PQ and not self.CP.openpilotLongitudinalControl:
      self.stopping = CS.acc_sys_stock["ACS_Anhaltewunsch"] and (CS.out.vEgoRaw <= 2 or self.stopping)
      self.stopped = self.EPB_enable and (CS.out.vEgoRaw == 0 or (self.stopping and self.stopped))

      if CS.acc_sys_stock["COUNTER"] != self.acc_sys_counter_last:
        EPB_handler(CS, self, CS.acc_sys_stock["ACS_Sta_ADR"], CS.acc_sys_stock["ACS_Sollbeschl"], CS.out.vEgoRaw, self.stopping)
        can_sends.append(self.CCS.filter_ACC_System(self.packer_pt, CANBUS.pt, CS.acc_sys_stock, self.EPB_active))
        can_sends.append(self.CCS.create_epb_control(self.packer_pt, CANBUS.body, self.EPB_brake, self.EPB_enable))
        can_sends.append(self.CCS.filter_epb1(self.packer_pt, CANBUS.cam, self.stopped))  # in custom module, filter the gateway fwd EPB msg
      if CS.acc_anz_stock["COUNTER"] != self.acc_anz_counter_last:
        can_sends.append(self.CCS.filter_ACC_Anzeige(self.packer_pt, CANBUS.pt, CS.acc_anz_stock, self.ACC_anz_blind))
      if self.frame % 2 or CS.motor2_stock != getattr(self, 'motor2_last', CS.motor2_stock):  # 50hz / 20ms
        can_sends.append(self.CCS.filter_motor2(self.packer_pt, CANBUS.cam, CS.motor2_stock, self.EPB_enable_history[0]))
        if CS.motor2_stock["GRA_Status"] in (1, 2) and self.motor2_last["GRA_Status"] == 0:
          self.EPB_enable_history = [False] * len(self.EPB_enable_history)  # disable filter when ECM enters cruise state
      if CS.bremse8_stock["COUNTER"] != self.bremse8_counter_last:
        can_sends.append(self.CCS.filter_bremse8(self.packer_pt, CANBUS.cam, CS.bremse8_stock, self.EPB_enable_history[0]))
      if CS.bremse11_stock["COUNTER"] != self.bremse11_counter_last:
        can_sends.append(self.CCS.filter_bremse11(self.packer_pt, CANBUS.cam, CS.bremse11_stock, self.stopped))
      if CS.gra_stock_values["COUNTER"] != self.gra_acc_counter_last:
        can_sends.append(self.CCS.filter_GRA_Neu(self.packer_pt, CANBUS.cam, CS.gra_stock_values, resume = self.stopped and (self.frame % 100 < 50)))

      self.motor2_last = CS.motor2_stock
      self.acc_sys_counter_last = CS.acc_sys_stock["COUNTER"]
      self.acc_anz_counter_last = CS.acc_anz_stock["COUNTER"]
      self.bremse8_counter_last = CS.bremse8_stock["COUNTER"]
      self.bremse11_counter_last = CS.bremse11_stock["COUNTER"]

    new_actuators = actuators.as_builder()
    new_actuators.steeringAngleDeg = self.apply_angle_last
    self.eps_timer_soft_disable_alert = False

    self.gra_acc_counter_last = CS.gra_stock_values["COUNTER"]
    self.frame += 1
    return new_actuators, can_sends
