use burn::record::PrecisionSettings;

use crate::burn::{Scope, TensorType, ToTokens, Type};
use super::{Node, NodeCodegen};
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct TopkNode {
    pub input: TensorType,
    pub output: TensorType,
}

// impl<PS: PrecisionSettings> NodeCodegen<PS> for TopkNode {
//     fn output_types(&self) -> Vec<Type> {
//         vec![Type::Tensor(self.output.clone())]
//     }

//     fn input_types(&self) -> Vec<Type> {
//         vec![Type::Tensor(self.input.clone())]
//     }

//     fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
//         let input = scope.tensor_use_owned(&self.input, node_position);
//         let output = &self.output.name;

//         let axes_arg = &self.axes.to_tokens();

//         quote! {
//             let #output = #input.squeeze_dims(&#axes_arg);
//         }
//     }
//     fn into_node(self) -> Node<PS> {
//         Node::Topk(self)
//     }
// }