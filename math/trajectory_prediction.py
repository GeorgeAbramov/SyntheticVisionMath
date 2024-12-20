import numpy as np
from typing import List
from equation_of_flight import FlightEquations
from RK4 import rk4_step

class TrajectoryPredictor:
    """
    Прогнозирование траектории полета
    """
    def __init__(self):
        self.equations = FlightEquations()
        
    def predict_trajectory(self, 
                         initial_state: np.ndarray,
                         dt: float,
                         prediction_time: float) -> List[np.ndarray]:
        """
        Прогноз траектории на заданное время
        
        Args:
            initial_state: начальное состояние [φ, λ, h, Vx, Vy, Vz, γ, θ, ψ]
            dt: шаг интегрирования
            prediction_time: время прогноза
            
        Returns:
            List[np.ndarray]: список точек траектории
        """
        steps = int(prediction_time / dt)
        trajectory = [initial_state.copy()]
        state = initial_state.copy()
        t = 0
        
        for _ in range(steps):
            state = rk4_step(
                derivatives=self.equations.calculate_derivatives,
                state=state,
                dt=dt,
                t=t
            )
            trajectory.append(state.copy())
            t += dt
            
        return trajectory