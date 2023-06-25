import inspect
import re


def extract_params(doc):
    if not doc:
        return {}
    param_pattern = r":param (\w+): (.+?)(?=:param|:return:|\Z)"  # added :return: as a stopping point
    params = re.findall(param_pattern, doc, re.DOTALL)
    param_dict = {}
    for name, desc in params:
        enum_pattern = r"<<(.*?)>>"
        enum_values = re.findall(enum_pattern, desc)
        if enum_values:
            enum_values = [value.strip() for value in enum_values[0].split(",")]
            param_dict[name] = {"description": re.sub(enum_pattern, '', desc).strip(), "enum": enum_values}
        else:
            param_dict[name] = {"description": desc.strip()}
    return param_dict

def type_mapping(dtype):
    if dtype == float:
        return "number"
    elif dtype == int:
        return "integer"
    elif dtype == str:
        return "string"
    else:
        return "string"


def func_to_json(func):
    func_name = func.__name__
    argspec = inspect.getfullargspec(func)
    func_doc = inspect.getdoc(func)
    func_description = re.split(r":param|:return:", func_doc)[0].strip() if func_doc else ""
    params = argspec.annotations.copy()
    if 'return' in params:
        del params['return']

    param_details = extract_params(func_doc)
    for param_name in argspec.args:
        params[param_name] = {
            "description": param_details.get(param_name, {}).get("description", ""),
            "type": type_mapping(params[param_name])
        }
        if "enum" in param_details.get(param_name, {}):
            params[param_name]["enum"] = param_details[param_name]["enum"]

    optional_params = argspec.defaults if argspec.defaults is not None else []
    len_optional_params = len(optional_params)
    return {
        "name": func_name,
        "description": func_description,
        "parameters": {
            "type": "object",
            "properties": params,
        },
        "required": argspec.args[:-len_optional_params] if len_optional_params else argspec.args,
    }

