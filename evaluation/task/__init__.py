from typing import List, Dict

from . import klue
from . import kobest
from . import kornlu
from . import kmmlu
from . import korquad
from . import korean_commongen
from .base import Task

TASK_REGISTRY = {
    "klue_nli": klue.NLI,
    "klue_sts": klue.STS,
    "klue_ynat": klue.YNAT,
    "klue_re": klue.RE,
    "kobest_boolq": kobest.BoolQ,
    "kobest_copa": kobest.COPA,
    "kobest_sentineg": kobest.SentiNeg,
    "kobest_hellaswag": kobest.HellaSwag,
    "kobest_wic": kobest.WiC,
    "kornlu_nli": kornlu.KorNLI,
    "kornlu_sts": kornlu.KorSTS,
    "kmmlu_nondestructive_testing": kmmlu.NondestructiveTesting,
    "kmmlu_math": kmmlu.Math,
    "kmmlu_chemistry": kmmlu.Chemistry,
    "kmmlu_industrial_engineer": kmmlu.IndustrialEngineer,
    "kmmlu_taxation": kmmlu.Taxation,
    "kmmlu_geomatics": kmmlu.Geomatics,
    "kmmlu_civil_engineering": kmmlu.CivilEngineering,
    "kmmlu_environmental_science": kmmlu.EnvironmentalScience,
    "kmmlu_chemical_engineering": kmmlu.ChemicalEngineering,
    "kmmlu_management": kmmlu.Management,
    "kmmlu_education": kmmlu.Education,
    "kmmlu_electronics_engineering": kmmlu.ElectronicsEngineering,
    "kmmlu_computer_science": kmmlu.ComputerScience,
    "kmmlu_food_processing": kmmlu.FoodProcessing,
    "kmmlu_gas_technology_and_engineering": kmmlu.GasTechnologyAndEngineering,
    "kmmlu_railway_and_automotive_engineering": kmmlu.RailwayAndAutomotiveEngineering,
    "kmmlu_fashion": kmmlu.Fashion,
    "kmmlu_agricultural_sciences": kmmlu.AgriculturalSciences,
    "kmmlu_biology": kmmlu.Biology,
    "kmmlu_marketing": kmmlu.Marketing,
    "kmmlu_patent": kmmlu.Patent,
    "kmmlu_realestate": kmmlu.RealEstate,
    "kmmlu_ecology": kmmlu.Ecology,
    "kmmlu_economics": kmmlu.Economics,
    "kmmlu_aviation_engineering_and_maintenance": kmmlu.AviationEngineeringAndMaintenance,
    "kmmlu_electrical_engineering": kmmlu.ElectricalEngineering,
    "kmmlu_korean_history": kmmlu.KoreanHistory,
    "kmmlu_mechanicalengineering": kmmlu.MechanicalEngineering,
    "kmmlu_telecommunications_and_wireless_technology": kmmlu.TelecommunicationsAndWirelessTechnology,
    "kmmlu_information_technology": kmmlu.InformationTechnology,
    "kmmlu_social_welfare": kmmlu.SocialWelfare,
    "kmmlu_energy_management": kmmlu.EnergyManagement,
    "kmmlu_construction": kmmlu.Construction,
    "kmmlu_materials_engineering": kmmlu.MaterialsEngineering,
    "kmmlu_maritime_engineering": kmmlu.MaritimeEngineering,
    "kmmlu_criminal_law": kmmlu.CriminalLaw,
    "kmmlu_political_science_and_sociology": kmmlu.PoliticalScienceAndSociology,
    "kmmlu_public_safety": kmmlu.PublicSafety,
    "kmmlu_health": kmmlu.Health,
    "kmmlu_machine_design_and_manufacturing": kmmlu.MachineDesignAndManufacturing,
    "kmmlu_accounting": kmmlu.Accounting,
    "kmmlu_law": kmmlu.Law,
    "kmmlu_psychology": kmmlu.Psychology,
    "kmmlu_refrigerating_machinery": kmmlu.RefrigeratingMachinery,
    "kmmlu_interior_architecture_and_design": kmmlu.InteriorArchitectureAndDesign,
    "korquad": korquad.KorQuad,
    "korquad_fewshot": korquad.KorQuadFewShot,
    "korean_commongen": korean_commongen.KoreanCommonGen,
}


def get_task_dict(data_dir: str, task_name_list: List[str]) -> Dict[str, Task]:
    """
    task_name 리스트에 맞는 Task 객체 리스트를 반환
    Args:
        data_dir: 데이터 디렉토리
        task_name_list: task_name 리스트 (e.g. ["klue_nli", "klue_sts"])
    Returns:
        task_name_dict: task_name을 key로 하는 Task 객체 딕셔너리
    """
    task_name_dict = {
        task_name: TASK_REGISTRY[task_name](data_dir=data_dir)
        for task_name in task_name_list
        if isinstance(task_name, str)
    }
    return task_name_dict
