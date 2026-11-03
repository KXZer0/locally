"""Small deterministic math tool. Interprets a fixed AST; never evals code."""
import ast
import math

def calculate(args):
    import sympy as sp
    expression = args.get('expression')
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 500:
        raise ValueError('Use a math expression of at most 500 characters.')
    symbols = {name:sp.Symbol(name) for name in ('x','y','z','t','n')}
    names = {**symbols,'pi':sp.pi,'e':sp.E}
    functions = {'sqrt':sp.sqrt,'sin':sp.sin,'cos':sp.cos,'tan':sp.tan,'log':sp.log,'exp':sp.exp,'abs':sp.Abs}
    budget = [64]
    def read(node, depth=0):
        budget[0] -= 1
        if budget[0] < 0 or depth > 12:
            raise ValueError('Expression is too complex.')
        if isinstance(node,ast.Constant) and type(node.value) in (int,float):
            if not math.isfinite(float(node.value)) or abs(node.value)>10**12:
                raise ValueError('Numeric literal is too large.')
            return sp.Rational(str(node.value))
        if isinstance(node,ast.Name) and node.id in names:
            return names[node.id]
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd)):
            value=read(node.operand,depth+1);return -value if isinstance(node.op,ast.USub) else value
        if isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Add,ast.Sub,ast.Mult,ast.Div,ast.Pow,ast.Mod)):
            left,right=read(node.left,depth+1),read(node.right,depth+1)
            if isinstance(node.op,ast.Pow):
                if not right.is_number or abs(right)>8:
                    raise ValueError('Powers must be numeric and between -8 and 8.')
                return left**right
            if isinstance(node.op,ast.Add):return left+right
            if isinstance(node.op,ast.Sub):return left-right
            if isinstance(node.op,ast.Mult):return left*right
            if isinstance(node.op,ast.Div):return left/right
            return left%right
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in functions and len(node.args)==1 and not node.keywords:
            argument=read(node.args[0],depth+1)
            if argument.is_number and abs(argument)>1000:
                raise ValueError('Function argument is too large.')
            return functions[node.func.id](argument)
        raise ValueError('Only numbers, basic math, x/y/z/t/n and supported math functions are allowed.')
    def parse(value):
        return read(ast.parse(value.strip(),mode='eval').body)
    if expression.count('=') > 1:raise ValueError('Use one equation.')
    parts=expression.replace('^','**').split('=')
    value=parse(parts[0])
    if len(parts)==2:value-=parse(parts[1])
    variable=args.get('solve_for')
    if len(parts)==2 or variable:
        if variable not in symbols:raise ValueError('Choose solve_for: x, y, z, t or n.')
        symbol=symbols[variable]
        if value.free_symbols-{symbol}:raise ValueError('Provide values for the other variables.')
        polynomial=sp.Poly(value,symbol)
        if polynomial.degree()>4:raise ValueError('Equation degree must be at most four.')
        return str(sp.solve(polynomial.as_expr(),symbol))
    if value.free_symbols:return str(value)
    if value.has(sp.zoo,sp.oo,-sp.oo,sp.nan):raise ValueError('The expression has no finite value.')
    return 'Exact: '+str(value)+'; decimal: '+str(sp.N(value,16))

CALCULATOR_TOOL={'type':'function','function':{'name':'calculator','description':'Evaluate exact arithmetic or solve a polynomial equation of degree at most four. Use this for reliable math. It accepts math expressions, not Python code.','parameters':{'type':'object','properties':{'expression':{'type':'string','description':'For example 17*23, sqrt(81), or 2*x+3=11.'},'solve_for':{'type':'string','enum':['x','y','z','t','n']}},'required':['expression'],'additionalProperties':False}}}
