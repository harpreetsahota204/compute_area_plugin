import os
import fiftyone as fo
from fiftyone.core.utils import add_sys_path
import fiftyone.operators as foo
from fiftyone.operators import types

from .utils import compute_areas

def _handle_calling(
        uri, 
        sample_collection, 
        field_name,
        computation_type,
        has_polylines,        
        delegate
        ):
    ctx = dict(dataset=sample_collection)

    params = dict(
        field_name,
        computation_type,
        has_polylines,
        delegate
        )
    return foo.execute_operator(uri, ctx, params=params)

class ComputeArea(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            # The operator's URI: f"{plugin_name}/{name}"
            name="compute_areas",  # required

            # The display name of the operator
            label="Compute Areas",  # required

            # A description for the operator
            description="Compute Bounding Box Area and Polygon Surface Area",

            icon="/assets/area-1-svgrepo-com.svg",
            )
    
    def resolve_input(self, ctx):
        """Implement this method to collect user inputs as parameters
        that are stored in `ctx.params`.

        Returns:
            a `types.Property` defining the form's components
        """
        inputs = types.Object()

        area_type = types.RadioGroup()
        area_type.add_choice("bbox_area", label="Compute Bounding Box Area")
        area_type.add_choice("surface_area", label="Compute Surface Area of a Polygon")        

        inputs.enum(
            "computation_type",
            values=area_type.values(),
            view=area_type,
            required=True,
            description="Select the type of area to compute:",
        )

        inputs.bool(
            "has_polylines",
            default=False,
            required=True,
            label="Are your segmentation masks represented as a FiftyOne Polylines?",
            description="If not, the operator will automatically convert the mask to Polylines and add it as a new Field to the Dataset.",
            
            )

        inputs.str(
            "field_name",
            label="Field name",
            required=True,
            description="Select a Field which contains either: Detections, Segmentations, or Polylines.",
        )

        inputs.bool(
            "delegate",
            default=False,
            required=True,
            label="Delegate execution?",
            description=("If you choose to delegate this operation you must first have a delegated service running. "
            "You can launch a delegated service by running `fiftyone delegated launch` in your terminal"),
            view=types.CheckboxView(),
        )

        inputs.view_target(ctx)

        return types.Property(inputs)

    def resolve_delegation(self, ctx):
        """Implement this method if you want to programmatically *force*
        this operation to be delegated or executed immediately.

        Returns:
            whether the operation should be delegated (True), run
            immediately (False), or None to defer to
            `resolve_execution_options()` to specify the available options
        """
        return ctx.params.get("delegate", False)


    def execute(self, ctx):
        """Executes the actual operation based on the hydrated `ctx`.
        All operators must implement this method.

        This method can optionally be implemented as `async`.

        Returns:
            an optional dict of results values
        """
        view = ctx.dataset
        field_name = ctx.params.get("field_name")
        computation_type= ctx.params.get("computation_type")
        has_polylines = ctx.params.get("has_polylines")
        
        # write main function here
        compute_areas(
            dataset= view, 
            field_name=field_name, 
            computation_type=computation_type,
            has_polylines=has_polylines
            )

        ctx.ops.reload_dataset()

    def __call__(
            self, 
            sample_collection, 
            field_name,
            computation_type,
            has_polylines,
            delegate
            ):
        return _handle_calling(
            self.uri,
            sample_collection,
            field_name,
            computation_type,
            has_polylines,
            delegate
            )

def register(p):
    """Always implement this method and register() each operator that your
    plugin defines.
    """
    p.register(ComputeArea)