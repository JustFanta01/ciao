# Versions of DuMeng Algorithm

- ### ```DuMeng1.py```  (does NOT work!)
    Algorithm as described in the paper in iteration (7).
The "integrator" and the unprojected dual variable have been merged into a single update that uses the differences in time i.e. $\lambda_k - \lambda_{k-1}$.

    $a_i^{k+1} = a_i^{k} - \gamma \sum_{j=0}^{N}r_{ij} a_{j}^k - \sum_{j=0}^{N}r_{ij}(\lambda_j^k - \lambda_j^{k-1}) + (\lambda_i^k - \lambda_i^{k-1}) + \beta B_i(z_i^{k+1} - z_i^{k})$


- ### ```DuMeng2.py``` (works!)
  Adapt (change notation and note that 'k+1' is 'i') and use the variable change as in Alghunaim DCPA paper i.e. $z_i = \mathcal L(x_i - \mathcal L y_i)$. Obtain the "network term" agent_i["nn"]. 
  
  $a_i^{k+1} = \lambda_i^{k} + \beta (B_i z_i^k - b_i) + n_i^k$
  
    But since that I shouldn't merge with the new variable states $k+1$ of my neighbors at iteration $k$ let's shift it back in time!

  From: $n_i^{k+1} = n_i^{k} - \mathcal{L}^2 (\lambda^{k+1} - \lambda^{k} + \gamma a^{k+1})$

  To: $n_i^{k+1} = n_i^{k} - \mathcal{L}^2 (\lambda^{k} - \lambda^{k-1} + \gamma a^{k})$

- ### ```DuMeng3.py``` (does NOT work!)
    try to be as close as possible to Alghunaim DCPA and use the new values of neighbors.
    code:```nn_k_plus_1[i] = agent_i["nn"] - (ll_k_plus_1 - ll_k + gamma * aa_k_plus_1).T @ rr_i.T```

- ### ```DuMeng4.py``` (does NOT work!)
    try to shift back trackers in time and match DuMeng3 structure.

- ### ```DuMeng5.py``` (working!)
    Explicitely describe the consensus term for the dual subsystem. 
    ```aa_k_plus_1[i] = agent_i["ll"] + beta * (B_i @ zz_k_plus_1[i] - b_i) + u```

    ```u``` is the control input of the dual gradient ascent driving the dissenus to zero.
    ```u = (K_P * u1) + (K_I * u2) + (K_D * u3)```
    ```u``` is structured as a PI (or PID) controller with:
    - $\text{error}_k \text{ = (setpoint - value)} = (0 - \mathcal L^2 \lambda_k)$
    - ```P-term```: $K_P \cdot \text{error}_k = K_P \cdot (0 )$, $K_P = 1$
    - ```I-term```: $K_I \cdot \text{integrator} = K_I \cdot y_{k+1}$
      with  $y_{k+1} = y_k + dt \cdot (\text{error}_k)$, $K_I = 1, dt = 1$
    - ```D-term```: $K_D \cdot \frac{1}{dt} (\text{error}_k - \text{error}_{k-1})$, $K_D = 0, dt = 1$


- ### ```DuMeng6.py```:
    Use Dynamic Average Consensus to track the $\lambda$ so maybe the proof will be simpler.
    It looks like it's working BUT IT IS EXTREMELY NOISY AND DOESN'T CONVERGE PRECISELY. 
    ERROR OF 10^(-2) ... --> so, it's not working