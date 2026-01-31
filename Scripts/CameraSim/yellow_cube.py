import omni.replicator.core as rep

mesh_path = "/World/Yellow_Cube"

with rep.get.prims(path_pattern=mesh_path):
    rep.modify.semantics([("class", "cube"), ("color", "yellow")])

