from src.abstract.explainer import ExplainerABC


def get_node_explainer(name: str)->ExplainerABC:
    """
    Returns the appropriate node explainer class based on the provided name.
    Parameters:
    name (str): The name of the explainer technique. Valid options are:
        - "cf-gnn": Returns CFExplainer
        - "cf-gnnfeatures": Returns CFExplainerFeatures
        - "random": Returns RandomExplainer
        - "random-feat": Returns RandomFeaturesExplainer
        - "ego": Returns EgoExplainer
        - "cff": Returns CFFExplainer
        - "unr": Returns UNRExplainer
        - "gnn-explainer": Raises NotImplemented error
    Returns:
    ExplainerABC: The corresponding explainer class for the given name.
    Raises:
    ValueError: If the provided name does not match any of the valid options.
    NotImplemented: If the provided name is "gnn-explainer".
    """

    if name == "cf-gnn":
        from src.node_level_explainer import CFExplainer
        return CFExplainer
    
    elif name == "cf-gnnfeatures":
        from src.node_level_explainer import CFExplainerFeatures
        return CFExplainerFeatures

    elif name == "random":
        from src.node_level_explainer import RandomExplainer
        return RandomExplainer

    elif name == "random-feat":
        from src.node_level_explainer import RandomFeaturesExplainer
        return RandomFeaturesExplainer
    
    elif name == "ego":
        from src.node_level_explainer import EgoExplainer
        return EgoExplainer
    
    elif name == "cff":
        from src.node_level_explainer import CFFExplainer
        return CFFExplainer
    
    elif name == "unr":
        from src.node_level_explainer import UNRExplainer
        return UNRExplainer
    
    elif name == "combined":
        from src.node_level_explainer import CombinedExplainer
        return CombinedExplainer
    
    elif name == "gnnexplainer":
        from src.node_level_explainer import GNNExplainerWrap
        return GNNExplainerWrap
    
    else:
        raise ValueError(f"Technique not implemented {name}")
    
    
    
def get_graph_explainer(name: str)->ExplainerABC:

    if name == "cf-gnnfeatures":
        from src.graph_level_explainer import CFExplainerFeatures
        return CFExplainerFeatures
    
    elif name == "cf-gnn":
        from src.graph_level_explainer import CFExplainer
        return CFExplainer
    
    elif name == "random":
        from src.graph_level_explainer import RandomExplainer
        return RandomExplainer

    elif name == "random-feat":
        from src.graph_level_explainer import RandomFeaturesExplainer
        return RandomFeaturesExplainer
    
    elif name == "ego":
        from src.graph_level_explainer import EgoExplainer
        return EgoExplainer
    
    elif name == "combined":
        from src.graph_level_explainer import CombinedExplainer
        return CombinedExplainer
    
    elif name == "cff":
        from src.graph_level_explainer import CFFExplainer
        return CFFExplainer
    
    elif name == "gnnexplainer":
        from src.graph_level_explainer import GNNExplainerWrap
        return GNNExplainerWrap
    
    else:
        raise ValueError(f"Technique not implemented {name}")