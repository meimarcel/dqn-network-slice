import numpy as np

from copy import deepcopy
from typing import List
from const import *


class Env:
    def __init__(self, fix_state=True):
        self.fix_state = fix_state
        self.cloud_capacity = 40_000
        self.fog_capacity = [20_000, 20_000]
        self.service_probability = np.array([4, 4, 2], dtype=np.float32) / np.sum([4, 4, 2])
        self.cloud_energy = 600
        self.fog_energy = 300

        self.lost_packets = 0
        self.energy_consumption = 0
        self.cloud_band = 0
        self.fog_band = [0, 0]
        self.cloud_active = False
        self.fog_active = [False, False]

        self.band_division = [{CLOUD: 1966, FOG: 0}, {CLOUD: 674.4, FOG: 1291.6}, {CLOUD: 74, FOG: 1892}]

        self.create_services()

    def create_services(self):
        self.total_ru_services = []

        if self.fix_state:
            self.total_ru_services = [
                [4, 1, 0],
                [3, 1, 1],
                [2, 1, 2],
                [0, 1, 4],
                [1, 1, 3],
                [0, 0, 5],
                [2, 2, 1],
                [3, 2, 0],
            ]
        else:
            for _ in range(8):
                radio_units = np.random.choice([EMBB, MMTC, URLLC], 5, p=self.service_probability)
                self.total_ru_services.append(
                    [
                        int((radio_units == EMBB).sum()),
                        int((radio_units == MMTC).sum()),
                        int((radio_units == URLLC).sum()),
                    ]
                )

    def reset(self):
        self.lost_packets = 0
        self.energy_consumption = 0
        self.cloud_band = 0
        self.fog_band = [0, 0]
        self.cloud_active = False
        self.fog_active = [False, False]

    def processing(self, time: int, split_quantity: List[int]):
        splits_number = deepcopy(split_quantity)
        current_ru_services = self.total_ru_services[time]
        self.lost_packets = 0

        cloud_band = deepcopy(self.cloud_band)
        fog_band = deepcopy(self.fog_band)
        cloud_active = deepcopy(self.cloud_active)
        fog_active = deepcopy(self.fog_active)
        energy_consumption = deepcopy(self.energy_consumption)

        for _ in range(current_ru_services[URLLC]):
            if splits_number[B] > 0:
                allocated_cloud = False
                allocated_fog = False

                if cloud_band < self.cloud_capacity:
                    if cloud_band + self.band_division[B][CLOUD] <= self.cloud_capacity:
                        cloud_band += self.band_division[B][CLOUD]
                        allocated_cloud = True

                    if not cloud_active:
                        cloud_active = True
                        energy_consumption += self.cloud_energy

                for i in range(len(fog_band)):
                    if fog_band[i] + self.band_division[B][FOG] <= self.fog_capacity[i]:
                        fog_band[i] += self.band_division[B][FOG]
                        allocated_fog = True

                        if not fog_active[i]:
                            fog_active[i] = True
                            energy_consumption += self.fog_energy

                        break

                if not allocated_cloud or not allocated_fog:
                    self.lost_packets += 1

                splits_number[B] -= 1
            else:
                if splits_number[I] > 0:
                    self.lost_packets += 1
                    splits_number[I] -= 1
                    continue

                if splits_number[E] > 0:
                    self.lost_packets += 1
                    splits_number[E] -= 1
                    continue

                self.lost_packets += 1

        for _ in range(current_ru_services[EMBB] + current_ru_services[MMTC]):
            if splits_number[E] > 0:
                allocated_cloud = False

                if cloud_band < self.cloud_capacity:
                    if cloud_band + self.band_division[E][CLOUD] <= self.cloud_capacity:
                        cloud_band += self.band_division[E][CLOUD]
                        allocated_cloud = True

                    if not cloud_active:
                        cloud_active = True
                        energy_consumption += self.cloud_energy

                if not allocated_cloud:
                    self.lost_packets += 1

                splits_number[E] -= 1
                continue

            if splits_number[I] > 0:
                allocated_cloud = False
                allocated_fog = False

                if cloud_band < self.cloud_capacity:
                    if cloud_band + self.band_division[I][CLOUD] <= self.cloud_capacity:
                        cloud_band += self.band_division[I][CLOUD]
                        allocated_cloud = True

                    if not cloud_active:
                        cloud_active = True
                        energy_consumption += self.cloud_energy

                for i in range(len(fog_band)):
                    if fog_band[i] + self.band_division[I][FOG] <= self.fog_capacity[i]:
                        fog_band[i] += self.band_division[I][FOG]
                        allocated_fog = True

                        if not fog_active[i]:
                            fog_active[i] = True
                            energy_consumption += self.fog_energy

                        break

                if not allocated_cloud or not allocated_fog:
                    self.lost_packets += 1

                splits_number[I] -= 1
                continue

            if splits_number[B] > 0:
                allocated_cloud = False
                allocated_fog = False

                if cloud_band < self.cloud_capacity:
                    if cloud_band + self.band_division[B][CLOUD] <= self.cloud_capacity:
                        cloud_band += self.band_division[B][CLOUD]
                        allocated_cloud = True

                    if not cloud_active:
                        cloud_active = True
                        energy_consumption += self.cloud_energy

                for i in range(len(fog_band)):
                    if fog_band[i] + self.band_division[B][FOG] <= self.fog_capacity[i]:
                        fog_band[i] += self.band_division[B][FOG]
                        allocated_fog = True

                        if not fog_active[i]:
                            fog_active[i] = True
                            energy_consumption += self.fog_energy

                        break

                if not allocated_cloud or not allocated_fog:
                    self.lost_packets += 1

                splits_number[B] -= 1
                continue

            self.lost_packets += 1

        if self.lost_packets == 0:
            self.cloud_band = cloud_band
            self.fog_band = fog_band
            self.cloud_active = cloud_active
            self.fog_active = fog_active
            self.energy_consumption = energy_consumption

            return [self.get_state(time), self.get_reward()] if time < 7 else [None, 100]
        else:
            return None, -100
    
    def get_reward(self):
        if self.energy_consumption == 600:
            return 1
        if self.energy_consumption == 900:
            return 0.5
        return 0.2

    def get_state(self, time=-1):
        if time == len(self.total_ru_services) - 1:
            services = [None, None, None]
        else:
            services = self.total_ru_services[time + 1]
        return deepcopy([self.cloud_band, *self.fog_band, *services])
