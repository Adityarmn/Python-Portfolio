def objective_function(Qpeak, Kpeak, Qgridlock, Kgridlock):
    # Buat rumus J berdasarkan Qpeak, Kpeak, Qgridlock, Kgridlock
    J = Qpeak/Kpeak + Qgridlock/Kgridlock
    return J