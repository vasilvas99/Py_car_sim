import math
from dataclasses import dataclass

# This simulation is based on a couple of assumptions - that acceleration/deceleration is directly proportional to the
# position of the gas pedal and the bicycle vehicle model, used to calculate the vehicle position and acceleration
# components, ref: https://thef1clan.com/2020/09/21/vehicle-dynamics-the-kinematic-bicycle-model/

MAX_ACCEL = 6  # m/s^2, taken from some high-end BMW model
MAX_DECEL = 10  # m/s^2, taken from some high-end BMW model
MAX_SPEED = 60  # m/s (approx 220 km/h)
DRAG_COEF = 0.  # 1/s

INTEGRATION_STEP = 1e-3  # s
CAR_LEN = 2  # m
LR = 1  # distance in m between the rear (non-steering) axle and the center of mass


@dataclass
class CarState:
    timestamp: float  # s
    gas_position: float  # dimensionless [-1,1]
    steer_angle: float  # radians [-pi/2, pi/2]
    heading_angle: float  # radians
    x_pos: float  # meters
    y_pos: float  # meters
    v_x: float  # m/s
    v_y: float  # m/s
    speed: float  # m/s
    accel_x: float  # m/s^2
    accel_y: float  # m/s^2
    accel_norm: float  # m/s^2


# "gas" position is between -1 and 1, -1 = max breaking, 1 = max acceleration
def gas_pos_to_acc(gas_pos):
    if gas_pos >= 0:
        return MAX_ACCEL * min(gas_pos, 1)
    else:
        return MAX_DECEL * max(gas_pos, -1)


def update_speed(speed_old, gas_pos):
    return min(max(0, speed_old + (gas_pos_to_acc(gas_pos) - DRAG_COEF * speed_old) * INTEGRATION_STEP), MAX_SPEED)


def beta(steering_angle):  # see ref doc
    return math.atan(LR * math.tan(steering_angle) / CAR_LEN)


def v_x(speed, heading, steering_angle):
    return speed * math.cos(heading + beta(steering_angle))


def v_y(speed, heading, steering_angle):
    return speed * math.sin(heading + beta(steering_angle))


def angular_velocity(speed, steering_angle):
    return speed * (math.tan(steering_angle) * math.cos(beta(steering_angle))) / CAR_LEN


def update_car(car: CarState, gas_pos: float, steer_angle: float) -> CarState:
    speed = update_speed(car.speed, gas_pos)
    heading_angle = car.heading_angle + angular_velocity(speed, steer_angle) * INTEGRATION_STEP
    vx = v_x(speed=speed, steering_angle=steer_angle, heading=heading_angle)
    vy = v_y(speed=speed, steering_angle=steer_angle, heading=heading_angle)
    x_pos = car.x_pos + vx * INTEGRATION_STEP
    y_pos = car.y_pos + vy * INTEGRATION_STEP
    timestamp = car.timestamp + INTEGRATION_STEP
    ax = (vx - car.v_x) / INTEGRATION_STEP
    ay = (vy - car.v_y) / INTEGRATION_STEP

    return CarState(
        timestamp=timestamp,
        gas_position=gas_pos,
        steer_angle=steer_angle,
        heading_angle=heading_angle,
        x_pos=x_pos,
        y_pos=y_pos,
        v_x=vx,
        v_y=vy,
        speed=speed,
        accel_x=ax,
        accel_y=ay,
        accel_norm=math.sqrt(ax ** 2 + ay ** 2)
    )


def main():
    car = CarState(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)  # stationary car at the origin
    while True:
        g = float(input("gas position -1 to 1 = "))
        s = float(input("steering input (rad) = "))
        car = update_car(car, g, s)
        print(f'{car}')


def test_data():
    car = CarState(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)  # stationary car at the origin
    for i in range(100_000):
        car = update_car(car, gas_pos=1, steer_angle=math.pi/4)
        print(f"{car.accel_norm}")


if __name__ == "__main__":
    test_data()
