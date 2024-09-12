import dspy

import random
import time

random.seed(time.time())

ALPHA = 0.1

BRANCHING_FACTOR = 2

BASE_PROMPT = ""

GOOD_THRESHOLD = 0.8

class OptimizationObject:
    def __init__(self, instructions="", created_by=[], score=-1, depth=-1, trace=None):
        self.created = []
        self.created_by = created_by
        self.instructions = instructions
        self.depth = depth
        self.score = score
        self.trace = trace
class Trace:
    def __init__(self, metaheuristic, heuristics):
        self.metaheuristic = metaheuristic
        self.heuristics = heuristics
class EvaluationRun:
    def __init__(self, iopairs, positive=False):
        self.iopairs = iopairs
        self.positive = positive
from typing import List, StrictBool, Optional, Tuple
from pydantic import BaseModel, Field

class AscendInput(BaseModel):
    iopairs: List[Tuple] = Field(description="A list of input output pairs produced by the system")
    positive: StrictBool = Field(description="Whether the output is a good one")

class AscendOutput(BaseModel):
    condition: str = Field(description="A condition which the system should look for")
    guidance: str = Field(description="A piece of guidance for when this situation occurs")

class AscendSignature(dspy.Signature):
    input: AscendInput = dspy.InputField()
    output: AscendOutput = dspy.OutputField()

class Ascend(dspy.Module):
    def __init__(self):
        self.generator = dspy.TypedChainOfThought(max_retries=15)
    def forward(self, input, depth):
        signature_instructions = "The system was given the following inputs and outputs." \ 
        "These outputs are of " + ("high" if input.positive else "low") + "quality. Attempt to notice a shared condition between" \
        "the inputs along with a piece of guidance the system should implement when the condition is true, to " \
        ("help the system continue to produce positive results" if input.positive else "stop the system from producing low quality results")
        self.generator.signature = AscendSignature.with_instructions(signature_instructions)
        output = self.generator(input=input)
        instructions = "IF:" + output.condition + "\n THEN: " + output.guidance + "\n"
        return OptimizationObject(instructions=instructions, depth=depth)

class DescendInput(BaseModel):
    metaheuristic: str = Field(description="The metaheuristic to guide the formation of new heuristics")
    heuristics: List[str] = Field(description="Lower level heuristics to extrapolate from")

class DescendOutput(BaseModel):
    condition_met: StrictBool = Field(description="Whether the heuristic's condition was met")
    new_heuristic: str = Field(description="A new piece of guidance if the metaheuristic's condition is met")

class DescendSignature(dspy.Signature):
    """Given the input of heuristics and a metaheuristic, determine whether the heuristics meet the condition of the metaheuristic, and if so, devise a new heuristic using the metaheuristic's guidance. Otherwise, return a null heuristic"""
    input: DescendInput = dspy.InputField()
    output: DescendOutput = dspy.OutputField()

class Descend(dspy.Module):
    def __init__(self):
        self.generator = dspy.TypedChainOfThought(signature=DescendSignature, max_retries=15)
    def forward(self, object, list_of_objects, priority_queue, base_prompt, evaluator, evaluation_module, metric):
        if(object.depth == 0):
            temp_signature = evaluation_module.signature
            evaluation_module.signature = evaluation_module.signature.__class__.with_instructions(evaluation_module.signature.instructions + "\n" + object.instructions)
            ev = evaluator(evaluation_module, metric=metric)
            list_of_objects 
        correct_leveled_objects = filter(list_of_objects, lambda x : x.depth == object.depth - 1)
        if len(correct_leveled_objects) < 2:
            pass
        for i in range(0, BRANCHING_FACTOR):
            heuristics = random.sample(correct_leveled_objects, 2)
            print("Generating new heuristic from metaheuristic")
            new_heuristic = self.generator(input=DescendInput(metaheuristic=object.instructions, heuristics=heuristics))
            if(new_heuristic.output.condition_met):
                new_object = OptimizationObject(instructions=new_heuristic.output.new_heuristic, depth=object.depth -1, trace=Trace(metaheuristic=object, heuristics=heuristics))
                new_object.created_by = [object]
                object.created.append(new_object)
                list_of_objects.append(new_object)

            else:
                i -= 1
class TreeOptimizer:
    def __init__(self, evaluator, metric, teacher_model, student_model, module, ):
        self.evaluator = evaluator
        self.metric = metric
        self.teacher_model = teacher_model
        self.student_model = student_model

        self.list_of_objects = []
        self.list_of_evaluation_runs = []
        self.base_prompt = module.signature.instructions
        random.seed(time.time())
    def run(self):
        pass
        



        


