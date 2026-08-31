use serde::{Deserialize, Serialize};
use crate::{Position3, Result, SdkError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all="snake_case")]
pub enum FormationType { Cluster, Line, Converging, Dispersing, Unknown }

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GroupMember { pub track_id:u64, pub position:Position3, pub velocity_m_per_s:[f64;3] }
impl GroupMember {
    pub fn new(track_id:u64, position:Position3, velocity_m_per_s:[f64;3])->Result<Self>{
        if velocity_m_per_s.iter().any(|v|!v.is_finite()){return Err(SdkError::InvalidArgument("group velocity must be finite".into()));}
        Ok(Self{track_id,position,velocity_m_per_s})
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GroupPerception {
    pub member_track_ids:Vec<u64>, pub formation:FormationType, pub coherence:f32,
    pub convergence_score:f32, pub fragmentation_score:f32, pub spatial_extent_m:f64,
}

pub fn perceive_group(members:&[GroupMember])->Result<GroupPerception>{
    if members.len()<2{return Err(SdkError::InvalidArgument("group perception requires at least two tracks".into()));}
    let n=members.len() as f64;
    let centroid=[members.iter().map(|m|m.position.x).sum::<f64>()/n,members.iter().map(|m|m.position.y).sum::<f64>()/n,members.iter().map(|m|m.position.z).sum::<f64>()/n];
    let mut conv=0.0; let mut frag=0.0; let mut extent=0.0;
    for m in members {
        let radial=[centroid[0]-m.position.x,centroid[1]-m.position.y,centroid[2]-m.position.z];
        let rnorm=norm(radial); let speed=norm(m.velocity_m_per_s);
        extent=extent.max(rnorm);
        if rnorm>1e-9 && speed>1e-9 {
            let alignment=dot(radial,m.velocity_m_per_s)/(rnorm*speed);
            conv+=alignment.max(0.0); frag+=(-alignment).max(0.0);
        }
    }
    conv/=n; frag/=n;
    let mut coherence_sum=0.0; let mut pairs=0.0;
    for i in 0..members.len(){for j in i+1..members.len(){let a=members[i].velocity_m_per_s;let b=members[j].velocity_m_per_s;let na=norm(a);let nb=norm(b);if na>1e-9&&nb>1e-9{coherence_sum+=(dot(a,b)/(na*nb)+1.0)*0.5;pairs+=1.0;}}}
    let coherence=if pairs>0.0{(coherence_sum/pairs).clamp(0.0,1.0)}else{0.0};
    let linearity=linearity_xy(members,centroid);
    let formation=if conv>0.65{FormationType::Converging}else if frag>0.65{FormationType::Dispersing}else if linearity>0.85{FormationType::Line}else if coherence>0.55{FormationType::Cluster}else{FormationType::Unknown};
    Ok(GroupPerception{member_track_ids:members.iter().map(|m|m.track_id).collect(),formation,coherence:coherence as f32,convergence_score:conv as f32,fragmentation_score:frag as f32,spatial_extent_m:extent})
}
fn dot(a:[f64;3],b:[f64;3])->f64{a[0]*b[0]+a[1]*b[1]+a[2]*b[2]}
fn norm(a:[f64;3])->f64{dot(a,a).sqrt()}
fn linearity_xy(members:&[GroupMember],c:[f64;3])->f64{let mut xx=0.0;let mut yy=0.0;let mut xy=0.0;for m in members{let x=m.position.x-c[0];let y=m.position.y-c[1];xx+=x*x;yy+=y*y;xy+=x*y;}let tr=xx+yy;if tr<=1e-9{return 0.0;}let disc=((xx-yy).powi(2)+4.0*xy*xy).sqrt();let l1=(tr+disc)*0.5;let l2=(tr-disc)*0.5;((l1-l2)/tr).clamp(0.0,1.0)}
