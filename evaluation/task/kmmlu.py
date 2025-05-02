import numpy as np
from .base import rf, Task
from .metrics import macro_f1_score, mean


class KMMLU(Task):
    VERSION = 0
    IN_HF_HUB = True
    DATASET_PATH = "bm_ko-MMLU"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["dev"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        out_doc = {
            "query": "{}".format(doc["question"]),
            "choices": [
                doc["A"],
                doc["B"],
                doc["C"],
                doc["D"],
            ],
            "gold": int(doc["answer"]) - 1,
        }
        return out_doc

    def doc_to_text(self, doc):
        return doc["query"]

    def doc_to_target(self, doc):
        return " " + doc["choices"][doc["gold"]]

    def construct_requests(self, doc, ctx):
        lls = [
            rf.loglikelihood(ctx, " {}".format(choice))[0] for choice in doc["choices"]
        ]

        return lls

    def process_results(self, doc, results):
        pred = np.argmax(results)
        gold = doc["gold"]

        acc = 1.0 if pred == gold else 0.0
        completion_len = np.array([float(len(i)) for i in doc["choices"]])
        acc_norm = 1.0 if np.argmax(results / completion_len) == gold else 0.0

        return {"acc": acc, "acc_norm": acc_norm, "macro_f1": (gold, pred)}

    def higher_is_better(self):
        return {"acc": True, "acc_norm": True, "macro_f1": True}

    def aggregation(self):
        return {"acc": mean, "acc_norm": mean, "macro_f1": macro_f1_score}


subjects = {
    "NondestructiveTesting": "Nondestructive-Testing",
    "Math": "math",
    "Chemistry": "Chemistry",
    "IndustrialEngineer": "Industrial-Engineer",
    "Taxation": "Taxation",
    "Geomatics": "Geomatics",
    "CivilEngineering": "Civil-Engineering",
    "EnvironmentalScience": "Environmental-Science",
    "ChemicalEngineering": "Chemical-Engineering",
    "Management": "Management",
    "Education": "Education",
    "ElectronicsEngineering": "Electronics-Engineering",
    "ComputerScience": "Computer-Science",
    "FoodProcessing": "Food-Processing",
    "GasTechnologyAndEngineering": "Gas-Technology-and-Engineering",
    "RailwayAndAutomotiveEngineering": "Railway-and-Automotive-Engineering",
    "Fashion": "Fashion",
    "AgriculturalSciences": "Agricultural-Sciences",
    "Biology": "Biology",
    "Marketing": "Marketing",
    "Patent": "Patent",
    "RealEstate": "Real-Estate",
    "Ecology": "Ecology",
    "Economics": "Economics",
    "AviationEngineeringAndMaintenance": "Aviation-Engineering-and-Maintenance",
    "ElectricalEngineering": "Electrical-Engineering",
    "KoreanHistory": "korean-history",
    "MechanicalEngineering": "Mechanical-Engineering",
    "TelecommunicationsAndWirelessTechnology": "Telecommunications-and-Wireless-Technology",
    "InformationTechnology": "Information-Technology",
    "SocialWelfare": "Social-Welfare",
    "EnergyManagement": "Energy-Management",
    "Construction": "Construction",
    "MaterialsEngineering": "Materials-Engineering",
    "MaritimeEngineering": "Maritime-Engineering",
    "CriminalLaw": "Criminal-Law",
    "PoliticalScienceAndSociology": "Political-Science-and-Sociology",
    "PublicSafety": "Public-Safety",
    "Health": "Health",
    "MachineDesignAndManufacturing": "Machine-Design-and-Manufacturing",
    "Accounting": "Accounting",
    "Law": "Law",
    "Psychology": "Psychology",
    "RefrigeratingMachinery": "Refrigerating-Machinery",
    "InteriorArchitectureAndDesign": "Interior-Architecture-and-Design",
}
for key, value in subjects.items():
    # 동적으로 클래스 생성
    globals()[key] = type(
        key,  # 클래스 이름
        (KMMLU,),  # 상속받을 클래스(KMMLU)
        {"DATASET_NAME": value},  # 클래스의 속성 정의
    )
