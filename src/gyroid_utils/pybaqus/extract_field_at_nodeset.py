from odbAccess import openOdb
import sys

# Abaqus CAE's field-output dropdown lists invariants ("Magnitude", "Mises", ...)
# alongside vector/tensor components ("U1", "S22", ...). Components are read via
# componentLabels + v.data[i]; invariants are precomputed attributes on each
# FieldValue instead, keyed here by the lowercased label CAE shows.
INVARIANT_ATTRS = {
    "magnitude": "magnitude",
    "mises": "mises",
    "tresca": "tresca",
    "pressure": "press",
    "third invariant": "inv3",
    "max principal": "maxPrincipal",
    "min principal": "minPrincipal",
    "mid principal": "midPrincipal",
    "max in-plane principal": "maxInPlanePrincipal",
    "min in-plane principal": "minInPlanePrincipal",
}

def extract_field_at_nodeset(odb_path:str, nset_name:str, field_name:str, out_path:str):
    # ---- open odb -------
    try:
        odb = openOdb(odb_path, readOnly=True)
    except Exception as e:
        print(f"Error opening ODB: {e}")
        return False
    assembly = odb.rootAssembly
    frame = odb.steps[odb.steps.keys()[-1]].frames[-1]   # last frame of last step

    # ------ get the node set --------
    try:
        nset = assembly.nodeSets[nset_name]
    except KeyError:
        print(f"Node set '{nset_name}' not found in the ODB.")
        odb.close()
        return False
    
    # ------ split "U.U1" style names into the field itself ("U") and the
    # component to pull out of it ("U1"); a bare name like "S" or "MISES"
    # has no component and is written out as-is --------
    if "." in field_name:
        base_field, component = field_name.split(".", 1)
    else:
        base_field, component = field_name, None

    # ------ extract the field output at the node set --------
    try:
        subset = frame.fieldOutputs[base_field].getSubset(region=nset)
    except KeyError:
        print(f"Field '{base_field}' not found in this frame. Available fields: {list(frame.fieldOutputs.keys())}")
        odb.close()
        return False
    except Exception as e:
        print(f"Error extracting field output: {e}")
        odb.close()
        return False

    # ------ resolve "component" to either an invariant attribute (e.g.
    # "Magnitude" -> v.magnitude) or an index into v.data (e.g. "U1" -> the
    # position of "U1" in componentLabels). componentLabels is empty for a
    # scalar field (v.data is then a single number, not a tuple).
    component_index = None
    invariant_attr = None
    if component is not None:
        invariant_attr = INVARIANT_ATTRS.get(component.lower())
        if invariant_attr is None:
            try:
                component_index = list(subset.componentLabels).index(component)
            except ValueError:
                print(f"Component '{component}' not found in field '{base_field}'. "
                      f"Available components: {list(subset.componentLabels)}, "
                      f"available invariants: {list(INVARIANT_ATTRS.keys())}")
                odb.close()
                return False

    # ------ write the extracted data to a CSV file, one value per node --------
    with open(out_path, 'w') as f:
        f.write("node_label,value\n")
        for v in subset.values:
            if invariant_attr is not None:
                value = getattr(v, invariant_attr)
            elif component_index is not None:
                value = v.data[component_index]
            else:
                value = v.data
            f.write("%d,%s\n" % (v.nodeLabel, value))

    odb.close()
    return True


if __name__ == "__main__":
    odb_path, nset_name, field_name, out_path = sys.argv[-4:]
    extract_field_at_nodeset(odb_path, nset_name, field_name, out_path)

