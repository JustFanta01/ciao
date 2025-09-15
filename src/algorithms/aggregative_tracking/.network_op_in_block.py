for k in range(max_iter):
    total_cost = 0
    total_grad = np.zeros(d)

    A = self.problem.adj  # (N,N) row-stocastica (meglio doppio-stocastica)

    # --- snapshot ---
    zz_prev = np.array([ag["zz"] for ag in self.problem.agents], dtype=float)  # (N,d)
    ss_prev = np.array([ag["ss"] for ag in self.problem.agents], dtype=float)  # (N,d)
    vv_prev = np.array([ag["vv"] for ag in self.problem.agents], dtype=float)  # (N,d)

    # grad locale + tracking μ (tutti da snapshot)
    nabla1 = np.array([ag.nabla_1(zz_prev[i], ss_prev[i]) for i, ag in enumerate(self.problem.agents)])
    # Se nabla_phi è una matrice (Jacobian), usa einsum; se è identità, è semplicemente vv_prev
    nabla_phi = np.array([ag.nabla_phi(zz_prev[i]) for i, ag in enumerate(self.problem.agents)])
    try:
        term = np.einsum('nij,ni->ni', nabla_phi, vv_prev)  # Jacobian(z_i) @ μ_i
    except ValueError:
        term = vv_prev  # fallback: caso scalare/identità

    grad = nabla1 + term
    
    zz_new = np.clip(zz_prev - stepsize * grad, 0, 1)

    # consenso su snapshot (nessuna dipendenza dall’ordine del loop)
    ss_cons = A @ ss_prev           # (N,N) @ (N,d) -> (N,d)
    phi_new  = np.array([ag.phi(zz_new[i])  for i, ag in enumerate(self.problem.agents)])
    phi_prev = np.array([ag.phi(zz_prev[i]) for i, ag in enumerate(self.problem.agents)])
    ss_new = ss_cons + (phi_new - phi_prev)

    vv_cons = A @ vv_prev
    nabla2_prev = np.array([ag.nabla_2(zz_prev[i], ss_prev[i]) for i, ag in enumerate(self.problem.agents)])
    nabla2_new  = np.array([ag.nabla_2(zz_new[i], ss_new[i])  for i, ag in enumerate(self.problem.agents)])
    vv_new = vv_cons + (nabla2_new - nabla2_prev)

    # [ write-back ]
    for i, ag in enumerate(self.problem.agents):
        ag["zz"] = zz_new[i]
        ag["ss"] = ss_new[i]
        ag["vv"] = vv_new[i]
    
    
    # [ collector ]
    collector_zz.log(k, np.array([ag["zz"] for ag in self.problem.agents]))
    collector_ss.log(k, np.array([ag["ss"] for ag in self.problem.agents]))
    collector_vv.log(k, np.array([ag["vv"] for ag in self.problem.agents]))

    collector_cost.log(k, total_cost)
    collector_grad.log(k, total_grad)