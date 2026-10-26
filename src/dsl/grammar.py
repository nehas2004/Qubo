"""
Lark Grammar specification for the Optimization Domain-Specific Language (DSL).
"""

DSL_GRAMMAR = r"""
    ?start: problem

    problem: [problem_name] var_section obj_section [con_section]

    problem_name: "PROBLEM" ":" CNAME

    var_section: "VARIABLES" ":" var_decl+

    var_decl: "binary" CNAME "[" INT "]"                             -> var_decl_binary_array
            | "binary" CNAME                                         -> var_decl_binary_scalar
            | "integer" CNAME "[" INT "]" "in" INT ".." INT          -> var_decl_int_array
            | "integer" CNAME "in" INT ".." INT                      -> var_decl_int_scalar

    obj_section: "OBJECTIVE" ":" OBJ_SENSE expr
    OBJ_SENSE: "MINIMIZE" | "minimize" | "MAXIMIZE" | "maximize"

    con_section: "CONSTRAINTS" ":" constraint+
    constraint: [CNAME ":"] expr COMP_OP expr                        -> constraint_def

    COMP_OP: "==" | "<=" | ">="

    ?expr: [SIGN] term (ADD_OP term)*

    ?term: factor (MUL_OP factor)*

    ?factor: NUMBER                                                  -> num
           | CNAME "[" INT "]"                                       -> var_indexed
           | CNAME                                                   -> var_scalar
           | "(" expr ")"

    ADD_OP: "+" | "-"
    MUL_OP: "*"
    SIGN: "+" | "-"

    COMMENT: /#[^\n]*/ | /\/\/[^\n]*/
    %ignore COMMENT

    %import common.CNAME
    %import common.INT
    %import common.NUMBER
    %import common.WS
    %ignore WS
"""
