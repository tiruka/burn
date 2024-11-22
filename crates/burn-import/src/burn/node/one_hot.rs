use super::{Node, NodeCodegen};
use crate::burn::{Scope, TensorType, ToTokens, Type};
use burn::config::Config;
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Config, Debug)]
pub struct OneHotConfig {
    pub depth: usize,
    pub axis: isize,
    pub on_value: f32,
    pub off_value: f32,
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
        let depth = self.config.depth.to_tokens();
        let on_value = self.config.on_value.to_tokens();
        let off_value = self.config.off_value.to_tokens();
        let axis = self.config.axis;

        quote! {
            let mut #output = #indices.one_hot(#depth);
            #output = #output * (#on_value - #off_value) + #off_value;
            if #axis != -1 {
                let output_shape = #output.shape();
                let rank = output_shape.len();
                let axis = if #axis < 0 {
                    (rank as isize + #axis) as usize
                } else {
                    #axis as usize
                };
                let mut permutation: Vec<usize> = (0..rank).collect();
                permutation.insert(axis, rank);
                #output = #output.permute(permutation);
            }
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
        node::{test::assert_tokens, one_hot::OneHotConfig, one_hot::OneHotNode},
        TensorType,
    };
    use burn::record::FullPrecisionSettings;

    #[test]
    fn test_codegen_one_hot() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();
        let config = OneHotConfig::new(3, -1, 1.0, 0.0);
        graph.register(OneHotNode::new(
            TensorType::new_int("indices", 2),
            TensorType::new_float("output", 3),
            config,
        ));
        graph.register_input_output(vec!["indices".to_string()], vec!["output".to_string()]);

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
                    let mut output = indices.one_hot(3);
                    output = output * (1 - 0) + 0;
                    if -1isize != -1 {
                        let output_shape = output.shape();
                        let rank = output_shape.len();
                        let axis = if -1isize < 0 {
                            (rank as isize + -1isize) as usize
                        } else {
                            -1isize as usize
                        };
                        let mut permutation: Vec<usize> = (0..rank).collect();
                        permutation.insert(axis, rank);
                        output = output.permute(permutation);
                    }
                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
