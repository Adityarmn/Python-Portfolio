import pandas as pd
from PSO import PSO
from pathlib import Path


class save_file:
    def __init__(
        self,
        out_xml_name: str,
        objective: int,
        Qpeak: int,
        Kpeak: int,
        Qgridlock: int,
        Kgridlock: int,
        density: int,  # Kt
        beta: int,  # beta
    ):
        self.out_xml_name = out_xml_name
        self.objective = objective
        self.Qpeak = Qpeak
        self.Kpeak = Kpeak
        self.Qgridlock = Qgridlock
        self.Kgridlock = Kgridlock
        self.density = density
        self.beta = beta
        self.episode = 0

    def iteration_episode(self, episode):
        if episode != 0:
            self.close()
            self.save_xml(self.out_xml_name, self.episode)
        self.episode += 1
        self.metrics = []

    def save_xml(self, out_xml_name, episode):
        """Save metrics of the simulation to a .xml file.

        Args:
        out_xml_name (str): Path to the output .xml file. E.g.: "results/my_results
        episode (int): Episode number to be appended to the output file name.
        """
        if out_xml_name is not None:
            df = pd.DataFrame(self.metrics)
            Path(Path(out_xml_name).parent).mkdir(parents=True, exist_ok=True)
            df.to_csv(
                out_xml_name + f"_conn{self.label}_ep{episode}" + ".xml", index=False
            )
            return pd.DataFrame(self.metrics)
