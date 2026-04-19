import sys
import numpy as np
import ast

def escape_latex(text):
    """Escapes LaTeX special characters to prevent errors."""
    if not isinstance(text, str):
        return str(text)
    special_chars = {
        '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
        '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}', '\\': r'\textbackslash{}'
    }
    return "".join(special_chars.get(c, c) for c in text)

def dict_to_tblr_only(data):
    """Generates a LaTeX tblr environment with gray15 header."""
    latex = [
        r"\begin{tblr}{",
        r"  colspec = {ll},",
        r"  hlines, vlines,",
        r"  row{1} = {font=\bfseries, bg=gray!15},",
        r"}",
        r"  Thông số& Giá trị\\"
    ]

    for key, value in data.items():
        key_escaped = escape_latex(str(key))

        # Handle numeric types (int, float, and all numpy numeric types)
        if isinstance(value, (int, float, np.number)):
            is_f = isinstance(value, (float, np.floating))
            if is_f and (abs(value) < 1e-3 or abs(value) > 1e5):
                formatted_val = f"{value:.4e}"
                if "e" in formatted_val:
                    base, exp = formatted_val.split("e")
                    # Format for LaTeX: 1.23 \times 10^{-264}
                    formatted_val = f"{base} \\times 10^{{{int(exp)}}}"
            elif is_f:
                formatted_val = f"{value:.4f}"
            else:
                formatted_val = str(value)
            val_escaped = f"${formatted_val}$"
        else:
            val_escaped = escape_latex(str(value))

        latex.append(f"  {key_escaped} & {val_escaped} \\\\")

    latex.append(r"\end{tblr}")
    return "\n".join(latex)

def main():
    # Check for required CLI arguments
    if len(sys.argv) < 3:
        print("Usage: python dict2tblr.py <input_file> <output_file>")
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        # General approach: Evaluate string in a sandboxed context
        # This allows 'np.float64' etc. to be recognized correctly
        safe_context = {
            "np": np,
            "numpy": np,
            "__builtins__": {} # Block dangerous functions
        }

        # Execute evaluation
        data = eval(content, safe_context)

        if not isinstance(data, dict):
            raise ValueError("Input data must be a dictionary.")

        # Generate LaTeX string
        latex_output = dict_to_tblr_only(data)

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex_output)

        print(f"Success! LaTeX table saved to: {output_path}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
