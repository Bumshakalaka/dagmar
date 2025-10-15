import re
from typing import Optional

from qdrant_client import models


def parse_filter_string(filter_str: str, allow_fields: list[str] | None = None) -> Optional[models.Filter]:
    """Parse a filter string into a Qdrant Filter object.

    Supported operators:
        - Equality: field=value
        - Inequality: field!=value
        - Null checking: field=null, field!=null
        - In list: field in [value1, value2, value3]
        - Not in list: field not in [value1, value2, value3]
        - Comparison: field>value, field<value, field>=value, field<=value
        - Text search: field like 'search text' (for TEXT indexed fields)
        - Logical: and, or, not
        - Grouping: parentheses ()

    Examples:
        - "Component=Hardware"
        - "Component=Hardware and Product=XYZ"
        - "linked_issues in [FAR-1, FAR-2, FAR-3]"
        - "(Component=Hardware or Component=Software) and Product=XYZ"
        - "Resolved>2024-01-01"
        - "not Component=Hardware"
        - "reported_failure like 'high current'"
        - "root_cause like 'supplier'"
        - "attachments=null"
        - "attachments!=null"
        - "root_cause!=null and Component=Hardware"

    Args:
        filter_str: Filter string to parse
        allow_fields: List of allowed field names. If provided, only conditions
                     for fields in this list will be included in the result.

    Returns:
        Qdrant Filter object or None if filter_str is None/empty

    """
    if not filter_str or not filter_str.strip():
        return None

    filter_str = filter_str.strip()

    # Helper function to check if a field is allowed
    def is_field_allowed(field_name: str) -> bool:
        """Check if a field is in the allow_fields list."""
        if allow_fields is None or len(allow_fields) == 0:
            return True
        return field_name in allow_fields

    # Helper function to parse a single condition
    def parse_condition(cond: str) -> Optional[models.Condition]:
        cond = cond.strip()

        # Handle "not in" operator - match from the right to handle field names with spaces
        # Pattern: field_name not in [val1, val2, ...]
        not_in_match = re.search(r"\s+not\s+in\s+\[(.+?)\]\s*$", cond, re.IGNORECASE)
        if not_in_match:
            # Extract field name (everything before " not in ")
            key = cond[: not_in_match.start()].strip()
            if not is_field_allowed(key):
                return None
            values_str = not_in_match.group(1).strip()
            values = [v.strip().strip("'\"") for v in values_str.split(",")]
            return models.FieldCondition(
                key=key,
                match=models.MatchExcept(**{"except": values}),  # type: ignore
            )

        # Handle "in" operator - match from the right to handle field names with spaces
        # Pattern: field_name in [val1, val2, ...]
        in_match = re.search(r"\s+in\s+\[(.+?)\]\s*$", cond, re.IGNORECASE)
        if in_match:
            # Extract field name (everything before " in ")
            key = cond[: in_match.start()].strip()
            if not is_field_allowed(key):
                return None
            values_str = in_match.group(1).strip()
            values = [v.strip().strip("'\"") for v in values_str.split(",")]
            return models.FieldCondition(
                key=key,
                match=models.MatchAny(any=values),
            )

        # Handle comparison operators (>=, <=, >, <)
        for op, qdrant_op in [(">=", "gte"), ("<=", "lte"), (">", "gt"), ("<", "lt")]:
            if op in cond:
                parts = cond.split(op, 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    if not is_field_allowed(key):
                        return None
                    value_str = parts[1].strip().strip("'\"")
                    # Try to convert to number
                    try:
                        value: float | str = float(value_str)
                        # Use numeric Range
                        range_params = {qdrant_op: value}
                        return models.FieldCondition(
                            key=key,
                            range=models.Range(**range_params),
                        )
                    except ValueError:
                        # Keep as string for dates and use DatetimeRange
                        # Qdrant accepts ISO format datetime strings
                        range_params = {qdrant_op: value_str}
                        return models.FieldCondition(
                            key=key,
                            range=models.DatetimeRange(**range_params),  # type: ignore
                        )

        # Handle != operator
        if "!=" in cond:
            parts = cond.split("!=", 1)
            if len(parts) == 2:
                key = parts[0].strip()
                if not is_field_allowed(key):
                    return None
                value = parts[1].strip().strip("'\"")

                # Handle !=null - this will be negated at the expression level
                # We'll return a special marker that the expression parser can handle
                if value.lower() == "null":
                    # Return a special condition that will be detected and negated
                    return models.IsEmptyCondition(
                        is_empty=models.PayloadField(key=key),
                    )
                # Try to convert to appropriate type
                if value.lower() in ("true", "false"):
                    value = value.lower() == "true"  # type: ignore
                elif value.isdigit():
                    value = int(value)  # type: ignore

                return models.FieldCondition(
                    key=key,
                    match=models.MatchExcept(**{"except": [value]}),  # type: ignore
                )

        # Handle like operator (text search for TEXT indexed fields)
        # Pattern: field_name like 'search text'
        like_match = re.search(r"\s+like\s+['\"](.+?)['\"]\s*$", cond, re.IGNORECASE)
        if like_match:
            # Extract field name (everything before " like ")
            key = cond[: like_match.start()].strip()
            if not is_field_allowed(key):
                return None
            search_text = like_match.group(1).strip()
            return models.FieldCondition(
                key=key,
                match=models.MatchText(text=search_text),
            )

        # Handle = operator (equality)
        if "=" in cond:
            parts = cond.split("=", 1)
            if len(parts) == 2:
                key = parts[0].strip()
                if not is_field_allowed(key):
                    return None
                value = parts[1].strip().strip("'\"")

                # Handle =null - use IsEmptyCondition
                if value.lower() == "null":
                    return models.IsEmptyCondition(
                        is_empty=models.PayloadField(key=key),
                    )
                # Try to convert to appropriate type
                if value.lower() in ("true", "false"):
                    value = value.lower() == "true"  # type: ignore
                elif value.isdigit():
                    value = int(value)  # type: ignore

                return models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value),
                )

        raise ValueError("Unable to parse condition: {cond}")

    # Recursive function to parse expressions with logical operators
    def parse_expression(expr: str) -> Optional[models.Filter]:
        expr = expr.strip()

        # Remove outer parentheses if they wrap the entire expression
        while expr.startswith("(") and expr.endswith(")"):
            # Check if these parens actually wrap the whole thing
            depth = 0
            wrapped = True
            for i, char in enumerate(expr[1:-1], 1):
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                if depth < 0:
                    wrapped = False
                    break
            if wrapped and depth == 0:
                expr = expr[1:-1].strip()
            else:
                break

        # Find logical operators at the top level (not inside parentheses)
        def find_top_level_operator(expr: str, operators: list) -> tuple | None:
            depth = 0
            for op in operators:
                if op == "not":
                    # Special handling for "not" to avoid matching "not in"
                    # Look for "not" that is NOT followed by " in "
                    pattern = r"^not\s+|(?<=\s)not\s+"
                else:
                    pattern = rf"\s+{op}\s+"

                for match in re.finditer(pattern, expr, re.IGNORECASE):
                    # Count parentheses depth up to this point
                    depth = 0
                    for char in expr[: match.start()]:
                        if char == "(":
                            depth += 1
                        elif char == ")":
                            depth -= 1

                    if depth == 0:
                        # For "not", check if it's followed by " in " (part of "not in")
                        if op == "not":
                            rest_of_expr = expr[match.end() :].lstrip()
                            if rest_of_expr.lower().startswith("in "):
                                continue  # Skip this "not" as it's part of "not in"
                        return (op, match.start(), match.end())
            return None

        # Look for OR first (lowest precedence)
        or_match = find_top_level_operator(expr, ["or"])
        if or_match:
            op, start, end = or_match
            left = expr[:start].strip()
            right = expr[end:].strip()
            left_filter = parse_expression(left)
            right_filter = parse_expression(right)

            # Filter out None values
            should_conditions = [f for f in [left_filter, right_filter] if f is not None]

            if len(should_conditions) == 0:
                return None
            elif len(should_conditions) == 1:
                return should_conditions[0]
            else:
                return models.Filter(should=should_conditions)  # type: ignore

        # Look for AND (higher precedence than OR)
        and_match = find_top_level_operator(expr, ["and"])
        if and_match:
            op, start, end = and_match
            left = expr[:start].strip()
            right = expr[end:].strip()
            left_filter = parse_expression(left)
            right_filter = parse_expression(right)

            # Combine must conditions, filtering out None values
            must_conditions = []
            if left_filter is not None:
                if hasattr(left_filter, "must") and left_filter.must:
                    must_conditions.extend(left_filter.must)
                else:
                    must_conditions.append(left_filter)

            if right_filter is not None:
                if hasattr(right_filter, "must") and right_filter.must:
                    must_conditions.extend(right_filter.must)
                else:
                    must_conditions.append(right_filter)

            if len(must_conditions) == 0:
                return None
            elif len(must_conditions) == 1:
                return models.Filter(must=must_conditions)
            else:
                return models.Filter(must=must_conditions)

        # Look for NOT (highest precedence)
        not_match = find_top_level_operator(expr, ["not"])
        if not_match:
            op, start, end = not_match
            rest = expr[end:].strip()
            rest_filter = parse_expression(rest)
            if rest_filter is None:
                return None
            return models.Filter(must_not=[rest_filter])

        # If no logical operators, parse as a single condition
        try:
            # Special handling for !=null patterns - these should be negated
            if re.search(r"!=\s*['\"]?null['\"]?\s*$", expr, re.IGNORECASE):
                # Parse the condition but wrap it in must_not
                condition = parse_condition(expr)
                if condition is None:
                    return None
                return models.Filter(must_not=[condition])
            else:
                condition = parse_condition(expr)
                if condition is None:
                    return None
                return models.Filter(must=[condition])
        except ValueError:
            raise ValueError("Failed to parse filter expression: {expr}")  # noqa: B904

    return parse_expression(filter_str)
