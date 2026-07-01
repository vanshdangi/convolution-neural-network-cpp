#include "layers/res_block.h"

ResidualBlock::ResidualBlock(int in_channels, int out_channels)
    :conv1(in_channels, out_channels, 3, 1),
    b1(out_channels),
    conv2(out_channels, out_channels, 3, 1),
    b2(out_channels),
    use_projection(in_channels != out_channels),
    shortcut(in_channels, out_channels, 1),
    shortcut_bn(out_channels)
{}

Tensor ResidualBlock::forward(const Tensor& x){

    Tensor out = conv1.forward(x);
    out = b1.forward(out);
    out = r1.forward(out);

    out = conv2.forward(out);
    out = b2.forward(out);

    const Tensor* identity = &x;
    Tensor projected;

    if (use_projection) {
        projected = shortcut.forward(x);
        projected = shortcut_bn.forward(projected);
        identity = &projected;
    }

    out += *identity;
    out = r2.forward(out);

    return out;
}

Tensor ResidualBlock::backward(const Tensor& grad_out){

    Tensor grad = r2.backward(grad_out);

    Tensor grad_residual = grad;
    Tensor grad_identity = grad;

    grad_residual = b2.backward(grad_residual);
    grad_residual = conv2.backward(grad_residual);
    grad_residual = r1.backward(grad_residual);
    grad_residual = b1.backward(grad_residual);
    grad_residual = conv1.backward(grad_residual);

    if(use_projection){
        grad_identity = shortcut_bn.backward(grad_identity);
        grad_identity = shortcut.backward(grad_identity);
    }

    return grad_residual + grad_identity;
}

void ResidualBlock::train() {
    b1.training = true;
    b2.training = true;
    if(use_projection){
        shortcut_bn.training = true;
    }
}

void ResidualBlock::eval() {
    b1.training = false;
    b2.training = false;
    if(use_projection){
        shortcut_bn.training = false;
    }
}
