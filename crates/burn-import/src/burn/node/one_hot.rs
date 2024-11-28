use super::{Node, NodeCodegen};
use crate::burn::{Scope, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct OneHotNode {
    pub indices: TensorType,
    pub depth: Type,
    pub values: TensorType,
    pub axis: isize,
    pub output: TensorType,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for OneHotNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn input_types(&self) -> Vec<Type> {
        vec![
            Type::Tensor(self.indices.clone()),
            self.depth.clone(),
            Type::Tensor(self.values.clone()),
        ]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let indices = scope.tensor_use_owned(&self.indices, node_position);
        let output = &self.output.name;
        let depth = match &self.depth {
            Type::Tensor(tensor) => scope.tensor_use_owned(tensor, node_position),
            Type::Scalar(scalar) => scalar.to_full_tensor(&[1]),
            _ => panic!(
                "OneHot depth needs Tensor or Scalar input, got {:?}!",
                self.depth
            ),
        };
        let values = scope.tensor_use_owned(&self.values, node_position);
        let axis = self.axis;
        let off_value = quote! { #values[0] };
        let on_value = quote! { #values[1] };
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
        node::{one_hot::OneHotNode, test::assert_tokens},
        TensorType,
    };
    use burn::record::FullPrecisionSettings;

    #[test]
    fn test_codegen_one_hot() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();
        graph.register(OneHotNode::new(
            TensorType::new_int("indices", 2),
            Type::Tensor(TensorType::new_float("depth", 1)),
            TensorType::new_float("values", 1),
            1,
            TensorType::new_float("output", 3),
        ));
        graph.register_input_output(
            vec![
                "indices".to_string(),
                "depth".to_string(),
                "values".to_string(),
            ],
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
                pub fn forward(
                    &self,
                    indices: Tensor<B, 2, Int>,
                    depth: Tensor<B, 1>,
                    values: Tensor<B, 1>,
                ) -> Tensor<B, 3> {
                    let mut output = indices.one_hot(depth);
                    output = output * (values[1] - values[0]) + values[0];
                    if 1isize != -1 {
                        let output_shape = output.shape();
                        let rank = output_shape.len();
                        let axis = if 1isize < 0 {
                            (rank as isize + 1isize) as usize
                        } else {
                            1isize as usize
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
