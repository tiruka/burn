use super::{Node, NodeCodegen};
use crate::burn::{Scope, TensorType, Type, TensorKind};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct OneHotConfig {
    axis: isize,
    depth: usize,
    values: Vec<i64>
}

#[derive(Debug, Clone, new)]
pub struct OneHotNode {
    pub indices: TensorType,
    pub output: TensorType,
    pub config: OneHotConfig,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for OneHotNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.indices.clone())]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let indices = scope.tensor_use_owned(&self.indices, node_position);
        let output = &self.output.name;
        let depth = &self.config.depth;
        let values = &self.config.values;
        let off_value = values[0];
        let on_value = values[1];
        let axis = self.config.axis;
        let output_expr = match &self.indices.kind {
            TensorKind::Int => {
                // Handle Int indices: directly use `indices.one_hot(depth)`
                quote! {
                    let mut #output = #indices.one_hot(#depth);
                }
            }
            TensorKind::Float => {
                // Handle Float indices: create a one-hot tensor manually
                quote! {
                    let mut #output = Tensor::<_, 1>::one_hot(#depth, #indices.shape()[0], &device);
                }
            }
            _ => panic!("Unsupported tensor type for indices: {:?}", self.indices),
        };
    
        let permute_expr = if axis != -1 {
            quote! {
                let output_shape = #output.shape();
                let rank = output_shape.dims.len();
                let axis = if #axis < 0 {
                    (rank as isize + #axis) as usize
                } else {
                    #axis as usize
                };
                let mut permutation: Vec<usize> = (0..rank).collect();
                permutation.insert(axis, rank);
                #output = #output.permute(permutation);
            }
        } else {
            quote! {}
        };
    
        quote! {
            #output_expr
            #output = #output * (#on_value - #off_value) + #off_value;
            #permute_expr
        }
    }

    fn into_node(self) -> super::Node<PS> {
        Node::OneHot(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burn::{
        graph::BurnGraph,
        node::{one_hot::OneHotNode, test::assert_tokens},
        TensorType,
    };
    use burn::record::FullPrecisionSettings;

    #[test]
    fn test_codegen_one_hot() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();
        let config = OneHotConfig::new(1, 1, vec![1, 1]);
        graph.register(OneHotNode::new(
            TensorType::new_int("indices", 2),
            TensorType::new_float("output", 3),
            config,
        ));
        graph.register_input_output(
            vec!["indices".to_string()],
            vec!["output".to_string()],
        );

        let expected = quote! {
            use burn::tensor::Int;
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };

            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model<B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }

                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, indices: Tensor<B, 2, Int>) -> Tensor<B, 3> {
                    let mut output = indices.one_hot(1usize);
                    output = output * (1i64 - 1i64) + 1i64;
                    let output_shape = output.shape();
                    let rank = output_shape.dims.len();
                    let axis = if 1isize < 0 {
                        (rank as isize + 1isize) as usize
                    } else {
                        1isize as usize
                    };
                    let mut permutation: Vec<usize> = (0..rank).collect();
                    permutation.insert(axis, rank);
                    output = output.permute(permutation);
                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
