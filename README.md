# pushing_tb_ws
The "turtlebot3_ws" in container 

 The main problems are:

## **🚨 CRITICAL FINDINGS:**

1. **Thread Proliferation Crisis**: Each robot has 32-33 threads instead of expected 7 (4-5x excess)
2. **Executor Saturation**: All robots running at 99-101% CPU due to thread thrashing
3. **Callback Starvation**: No subscribers detected for critical topics (even though topics have data)
4. **Thread Contention**: High CPU + excessive threads = thrashing/blocking

solution: shared callback group 

problem: PickUp second time launch, shows: [PickingUp] MPC control step 2/3: v=-0.010, ω=0.001, dist_to_target=0.048m, dist_to_traj=0.048m, traj_idx=483/484 