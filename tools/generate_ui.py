from utils.langfuse_model_wrapper import langfuse_model_wrapper
from e2b_code_interpreter import CodeInterpreter, Result
from langchain.pydantic_v1 import BaseModel
from langchain.tools import BaseTool
from guardrails.hub import ValidPython
from guardrails import Guard
from langfuse import Langfuse
from typing import Type, Tuple
from eezo.interface.message import Message
from eezo import Eezo

import json

l = Langfuse()
e = Eezo()

generate_ui = l.get_prompt("generate_ui")
# Install with: guardrails hub install hub://reflex/valid_python
guard = Guard().use(ValidPython, on_fail="exception")


class GenerateUIArgsSchema(BaseModel):
    query: str


class GenerateUI(BaseTool):
    name = "Generate UI"
    description = "Generate a UI for a given context."
    args_schema: Type[BaseModel] = GenerateUIArgsSchema
    message: Message | None
    input_str: str | None

    def __init__(
        self, message: Message | None = None, input_str: str = "No history privided."
    ):
        super().__init__()
        self.message = message
        self.input_str = input_str

    def _code_interpret(
        self, code_interpreter: CodeInterpreter, code: str
    ) -> Tuple[Result, str]:
        print(f"\n{'='*50}\n> Running following AI-generated code:\n{code}\n{'='*50}")
        exec = code_interpreter.notebook.exec_cell(
            code,
            # You can stream logs from the code interpreter
            # on_stderr=lambda stderr: print("\n[Code Interpreter stdout]", stderr),
            # on_stdout=lambda stdout: print("\n[Code Interpreter stderr]", stdout),
            #
            # You can also stream additional results like charts, images, etc.
            # on_result=...
        )

        if exec.error:
            print("[Code Interpreter error]", exec.error)  # Runtime error
        else:
            return exec.results[0], exec.logs

    def _run(self, **kwargs):
        system_prompt = generate_ui.compile(
            input_str=self.input_str, user_prompt=kwargs["query"]
        )

        result = langfuse_model_wrapper(
            name="GenerateUI",
            system_prompt=system_prompt,
            prompt=generate_ui,
            user_prompt=kwargs["query"],
            model="gpt-4o",
            temperature=0.7,
        )

        code_str = result.split("```python\n")[1].split("\n```")[0]

        try:
            # https://hub.guardrailsai.com/validator/reflex/valid_python
            # If you get this error, upggade pip
            #   [  ==] Downloading dependenciesERROR: unknown command "inspect"
            #   Failed to inspect
            #   Exit code: 1
            # stdout: b''
            guard.validate(code_str)

            # Execute the code
            print(
                f"\n{'='*50}\n> Running following AI-generated code:\n{code_str}\n{'='*50}"
            )

            ui_components = []
            with CodeInterpreter() as code_interpreter:
                code_result, _ = self._code_interpret(code_interpreter, code_str)
                json_str = code_result.raw["text/plain"]
                cleaned_json_str = json_str.strip("'").replace("\\\\", "\\")
                ui_components = json.loads(cleaned_json_str)

            for component in ui_components:
                self.message.add(component["type"], **component["props"])
            self.message.notify()
        except Exception as e:
            self.message.add("text", text="An error occurred while executing the code:")
            code_str = f"Error: {e}"
            self.message.add("text", text=code_str)
            self.message.notify()

        return code_str
