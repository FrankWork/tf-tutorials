#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

REGISTER_OP("ExampleOp")
    .Input("x: float")
    .Output("y: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
      })
    .SetIsStateful()
    .Doc(R"doc(GPU op Example)doc");

}  // end namespace tensorflow
