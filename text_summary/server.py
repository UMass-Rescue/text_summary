from typing import TypedDict

from flask_ml.flask_ml_server import MLServer
from flask_ml.flask_ml_server.models import (
    InputSchema,
    InputType,
    ParameterSchema,
    EnumParameterDescriptor,
    ResponseBody,
    TaskSchema,
    EnumVal,
    TextResponse,
    DirectoryInput,
)
from text_summary.model import SUPPORTED_MODELS
from text_summary.summarize import process_files
import json


class Inputs(TypedDict):
    input_dir: DirectoryInput
    output_dir: DirectoryInput


class Parameters(TypedDict):
    model: str


def task_schema() -> TaskSchema:
    input_dir_schema = InputSchema(
        key="input_dir",
        label="Path to the directory containing the input files",
        input_type=InputType.DIRECTORY,
    )
    output_dir_schema = InputSchema(
        key="output_dir",
        label="Path to the directory containing the output files",
        input_type=InputType.DIRECTORY,
    )
    parameter_schema = ParameterSchema(
        key="model",
        label="Model to use for summarization",
        subtitle="Model to use for summarization",
        value=EnumParameterDescriptor(
            enum_vals=[EnumVal(key=model, label=model) for model in SUPPORTED_MODELS],
            default=SUPPORTED_MODELS[0],
        ),
    )
    return TaskSchema(
        inputs=[input_dir_schema, output_dir_schema], parameters=[parameter_schema]
    )


server = MLServer(__name__)
server.add_app_metadata(
    name="Text Summarization",
    author="UMass Rescue",
    version="0.1.0",
    info="Summarize text and PDF files in a directory.",
)


@server.route(
    "/summarize", task_schema_func=task_schema, short_title="Summarize Text", order=0
)
def summarize(
    inputs: Inputs,
    parameters: Parameters,
) -> ResponseBody:
    """
    Summarize text and PDF files in a directory.
    """
    input_dir = inputs["input_dir"].path
    output_dir = inputs["output_dir"].path
    model = parameters["model"]

    processed_files = process_files(model, input_dir, output_dir)

    response = TextResponse(value=json.dumps(list(processed_files)))
    return ResponseBody(root=response)


if __name__ == "__main__":
    # Run a debug server
    server.run()
