use std::collections::HashSet;

use crate::{
    scene::{ColorSpace, Image},
    *,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(crate = "serde")]
pub enum ShaderKind {
    #[serde(rename = "surface")]
    Surface,
    #[serde(rename = "volume")]
    Volume,
    #[serde(rename = "emission")]
    Emission,
    #[serde(rename = "background")]
    Background,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(crate = "serde")]
pub struct ShaderGraph {
    pub nodes: Collection<ShaderNode>,
    pub output: NodeRef<ShaderNode>,
    pub kind: ShaderKind,
}
impl<'a> Index<&'a NodeRef<ShaderNode>> for ShaderGraph {
    type Output = ShaderNode;
    fn index(&self, index: &'a NodeRef<ShaderNode>) -> &Self::Output {
        &self.nodes[index]
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(crate = "serde")]
pub enum MathOp {
    #[serde(rename = "add")]
    Add,
    #[serde(rename = "sub")]
    Sub,
    #[serde(rename = "mul")]
    Mul,
    #[serde(rename = "div")]
    Div,
    #[serde(rename = "pow")]
    Pow,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(crate = "serde")]
pub enum BsdfPreference {
    #[serde(rename = "mix")]
    LinearMix, //linear mix of lobes
    #[serde(rename = "pfmc")]
    PositionFreeMc, //layered material via position-free monte carlo
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(crate = "serde")]
pub struct PrincipledBsdf {
    pub base_color: NodeRef<ShaderNode>,
    pub metallic: NodeRef<ShaderNode>,
    pub roughness: NodeRef<ShaderNode>,
    pub ior: NodeRef<ShaderNode>,
    pub alpha: NodeRef<ShaderNode>,
    pub normal: NodeRef<ShaderNode>,
    pub subsurface_weight: NodeRef<ShaderNode>,
    pub subsurface_radius: NodeRef<ShaderNode>,
    pub subsurface_scale: NodeRef<ShaderNode>,
    // pub subsurface_ior: NodeRef<ShaderNode>,
    pub subsurface_anisotropy: NodeRef<ShaderNode>,
    pub specular_ior_level: NodeRef<ShaderNode>,
    pub specular_tint: NodeRef<ShaderNode>,
    pub anisotropic: NodeRef<ShaderNode>,
    pub anisotropic_rotation: NodeRef<ShaderNode>,
    pub tangent: NodeRef<ShaderNode>,
    pub transmission_weight: NodeRef<ShaderNode>,
    pub sheen_weight: NodeRef<ShaderNode>,
    pub sheen_tint: NodeRef<ShaderNode>,
    pub coat_weight: NodeRef<ShaderNode>,
    pub coat_roughness: NodeRef<ShaderNode>,
    pub coat_ior: NodeRef<ShaderNode>,
    pub coat_tint: NodeRef<ShaderNode>,
    pub coat_normal: NodeRef<ShaderNode>,
    pub emission_color: NodeRef<ShaderNode>,
    pub emission_strength: NodeRef<ShaderNode>,
    pub preference: BsdfPreference,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Hash, PartialEq, Eq)]
#[serde(crate = "serde")]
pub enum NormalMapSpace {
    #[serde(rename = "tangent")]
    TangentSpace,
    #[serde(rename = "object")]
    ObjectSpace,
    #[serde(rename = "world")]
    WorldSpace,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Hash, PartialEq, Eq)]
#[serde(crate = "serde")]
pub enum MappingType {
    #[serde(rename = "point")]
    Point,
    #[serde(rename = "texture")]
    Texture,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Hash, PartialEq, Eq)]
#[serde(crate = "serde")]
pub enum SeparateColorMode {
    #[serde(rename = "rgb")]
    Rgb,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(crate = "serde")]
#[serde(tag = "type")]
pub enum ShaderNode {
    #[serde(rename = "float")]
    Float { value: f32 },
    #[serde(rename = "float3")]
    Float3 { value: [f32; 3] },
    #[serde(rename = "float4")]
    Float4 { value: [f32; 4] },
    #[serde(rename = "math")]
    Math {
        op: MathOp,
        first: NodeRef<ShaderNode>,
        second: NodeRef<ShaderNode>,
    },
    #[serde(rename = "rgb")]
    Rgb {
        value: [f32; 3],
        colorspace: ColorSpace,
    },
    #[serde(rename = "image")]
    TexImage {
        image: Image,
        #[serde(default)]
        uv: Option<NodeRef<ShaderNode>>,
    },
    #[serde(rename = "noise")]
    PerlinNoise {
        dim: u32,
        scale: NodeRef<ShaderNode>,
    },
    #[serde(rename = "mapping")]
    Mapping {
        vector: NodeRef<ShaderNode>,
        mapping: MappingType,
        location: NodeRef<ShaderNode>,
        rotation: NodeRef<ShaderNode>,
        scale: NodeRef<ShaderNode>,
    },
    #[serde(rename = "diffuse")]
    DiffuseBsdf { color: NodeRef<ShaderNode> },
    #[serde(rename = "metal")]
    MetalBsdf {
        eta: String,
        roughness: NodeRef<ShaderNode>,
    },
    #[serde(rename = "plastic")]
    PlasticBsdf {
        kd: NodeRef<ShaderNode>,
        ks: NodeRef<ShaderNode>,
        eta: NodeRef<ShaderNode>,
        roughness: NodeRef<ShaderNode>,
    },
    #[serde(rename = "glass")]
    GlassBsdf {
        color: NodeRef<ShaderNode>,
        ior: NodeRef<ShaderNode>,
        roughness: NodeRef<ShaderNode>,
    },
    #[serde(rename = "spectral_uplift")]
    SpectralUplift { rgb: NodeRef<ShaderNode> },
    #[serde(rename = "principled")]
    PrincipledBsdf {
        #[serde(flatten)]
        bsdf: Box<PrincipledBsdf>,
    },
    #[serde(rename = "checkerboard")]
    Checkerboard {
        vector: Option<NodeRef<ShaderNode>>,
        scale: NodeRef<ShaderNode>,
        color1: NodeRef<ShaderNode>,
        color2: NodeRef<ShaderNode>,
    },
    #[serde(rename = "normal_map")]
    NormalMap {
        normal: NodeRef<ShaderNode>,
        strength: NodeRef<ShaderNode>,
        space: NormalMapSpace,
    },
    #[serde(rename = "texcoords")]
    TexCoords {},
    #[serde(rename = "emission")]
    Emission {
        color: NodeRef<ShaderNode>,
        strength: NodeRef<ShaderNode>,
    },
    #[serde(rename = "mix")]
    MixBsdf {
        first: NodeRef<ShaderNode>,
        second: NodeRef<ShaderNode>,
        factor: NodeRef<ShaderNode>,
    },
    #[serde(rename = "extract")]
    Extract {
        node: NodeRef<ShaderNode>,
        field: String,
    },
    #[serde(rename = "separate_color")]
    SeparateColor {
        mode: SeparateColorMode,
        color: NodeRef<ShaderNode>,
    },
    #[serde(rename = "output")]
    Output { node: NodeRef<ShaderNode> },
}
pub trait NodeVisitor: Sized {
    fn visit_mut(&mut self, graph: &mut ShaderGraph, node: &NodeRef<ShaderNode>) {
        let node = graph[node].clone();
        match node {
            ShaderNode::Float { .. } => {}
            ShaderNode::Float3 { .. } => {}
            ShaderNode::Float4 { .. } => {}
            ShaderNode::Math {
                op: _,
                first,
                second,
            } => {
                graph.visit_mut(self, &first);
                graph.visit_mut(self, &second);
            }
            ShaderNode::Rgb { .. } => {}
            ShaderNode::TexImage { .. } => {}
            ShaderNode::PerlinNoise { dim: _, scale } => {
                graph.visit_mut(self, &scale);
            }
            ShaderNode::DiffuseBsdf { color } => {
                graph.visit_mut(self, &color);
            }
            ShaderNode::GlassBsdf {
                color,
                ior,
                roughness,
            } => {
                graph.visit_mut(self, &color);
                graph.visit_mut(self, &ior);
                graph.visit_mut(self, &roughness);
            }
            ShaderNode::SpectralUplift { rgb } => {
                graph.visit_mut(self, &rgb);
            }
            ShaderNode::PrincipledBsdf { bsdf } => {
                graph.visit_mut(self, &bsdf.base_color);
                graph.visit_mut(self, &bsdf.metallic);
                graph.visit_mut(self, &bsdf.roughness);
                graph.visit_mut(self, &bsdf.ior);
                graph.visit_mut(self, &bsdf.alpha);
                graph.visit_mut(self, &bsdf.normal);
                graph.visit_mut(self, &bsdf.subsurface_weight);
                graph.visit_mut(self, &bsdf.subsurface_radius);
                graph.visit_mut(self, &bsdf.subsurface_scale);
                // graph.visit_mut(self, &bsdf.subsurface_ior);
                graph.visit_mut(self, &bsdf.subsurface_anisotropy);
                graph.visit_mut(self, &bsdf.specular_ior_level);
                graph.visit_mut(self, &bsdf.specular_tint);
                graph.visit_mut(self, &bsdf.anisotropic);
                graph.visit_mut(self, &bsdf.anisotropic_rotation);
                graph.visit_mut(self, &bsdf.tangent);
                graph.visit_mut(self, &bsdf.transmission_weight);
                graph.visit_mut(self, &bsdf.sheen_weight);
                graph.visit_mut(self, &bsdf.sheen_tint);
                graph.visit_mut(self, &bsdf.coat_weight);
                graph.visit_mut(self, &bsdf.coat_roughness);
                graph.visit_mut(self, &bsdf.coat_ior);
                graph.visit_mut(self, &bsdf.coat_tint);
                graph.visit_mut(self, &bsdf.coat_normal);
                graph.visit_mut(self, &bsdf.emission_color);
                graph.visit_mut(self, &bsdf.emission_strength);
            }

            ShaderNode::Emission { color, strength } => {
                graph.visit_mut(self, &color);
                graph.visit_mut(self, &strength);
            }
            ShaderNode::MixBsdf {
                first,
                second,
                factor,
            } => {
                graph.visit_mut(self, &first);
                graph.visit_mut(self, &second);
                graph.visit_mut(self, &factor);
            }
            ShaderNode::Extract { node, field: _ } => {
                graph.visit_mut(self, &node);
            }
            ShaderNode::Output { node } => {
                graph.visit_mut(self, &node);
            }
            _ => {}
        }
    }
}

impl ShaderGraph {
    pub fn visit_mut<V: NodeVisitor>(&mut self, visitor: &mut V, node: &NodeRef<ShaderNode>) {
        visitor.visit_mut(self, node);
    }

    pub fn remove_unused(&mut self) {
        struct Reachable {
            reachable: HashSet<NodeRef<ShaderNode>>,
        }
        impl NodeVisitor for Reachable {
            fn visit_mut(&mut self, graph: &mut ShaderGraph, node: &NodeRef<ShaderNode>) {
                self.reachable.insert(node.clone());
                graph.visit_mut(self, node)
            }
        }
        let mut reachable = Reachable {
            reachable: HashSet::new(),
        };
        self.visit_mut(&mut reachable, &self.output.clone());
        let mut to_remove = Vec::new();
        for (node, _) in self.nodes.iter() {
            if !reachable.reachable.contains(&node) {
                to_remove.push(node.clone());
            }
        }
        for node in to_remove {
            self.nodes.remove(&node);
        }
    }
}
