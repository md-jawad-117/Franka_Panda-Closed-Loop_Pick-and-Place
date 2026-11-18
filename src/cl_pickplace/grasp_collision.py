"""A separate CAD query world constructed without live scene state.

MuJoCo's convex collision meshes are copied from robot CAD, not from a scene
observation. The only target poses are explicit learned inputs. No dynamics or
successful-grasp oracle is used in this world.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
import mujoco
import numpy as np

from .grasp_planner import fixture_boxes
from .kinematics import mat_to_quat


def values(array):return ' '.join(format(float(x),'.12g') for x in np.asarray(array).ravel())


def extract_robot_cad(model, calibration):
    d=mujoco.MjData(model);d.qpos[calibration['arm_qadr']]=calibration['home_q']
    d.qpos[calibration['finger_qadr']]=0.;mujoco.mj_kinematics(model,d)
    tcp=d.site_xpos[calibration['ee_site']];rot=d.site_xmat[calibration['ee_site']].reshape(3,3)
    bodies={calibration['hand_body']:'hand',calibration['left_body']:'left',calibration['right_body']:'right'}
    rows=[]
    for g in range(model.ngeom):
        body=int(model.geom_bodyid[g])
        if body not in bodies or not (model.geom_contype[g] or model.geom_conaffinity[g]):continue
        gr=d.geom_xmat[g].reshape(3,3)
        row={'source_geometry_id':g,'body':bodies[body],'position_m':((d.geom_xpos[g]-tcp)@rot).tolist(),
             'quaternion':mat_to_quat(rot.T@gr).tolist()}
        if model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH:
            mesh=int(model.geom_dataid[g]);v=int(model.mesh_vertadr[mesh]);n=int(model.mesh_vertnum[mesh])
            f=int(model.mesh_faceadr[mesh]);nf=int(model.mesh_facenum[mesh])
            row.update(type='mesh',vertices=model.mesh_vert[v:v+n].tolist(),faces=model.mesh_face[f:f+nf].tolist())
        elif model.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX:row.update(type='box',half_size=model.geom_size[g].tolist())
        else:raise ValueError('Unsupported robot collision geometry')
        rows.append(row)
    return rows


class GraspCollisionWorld:
    """Independent geometric queries. Live simulation data is never accepted."""
    def __init__(self,robot_cad,fixture_cad,table_z,capsule_cad,fixture_pose,inflation=0.):
        root=ET.Element('mujoco',model='learned_pose_cad_queries')
        ET.SubElement(root,'compiler',angle='radian')
        ET.SubElement(root,'option',gravity='0 0 0')
        asset=ET.SubElement(root,'asset');world=ET.SubElement(root,'worldbody')
        tcp=ET.SubElement(world,'body',name='tcp');ET.SubElement(tcp,'freejoint',name='tcp_joint')
        bodies={'hand':tcp}
        for side,axis in (('left','0 1 0'),('right','0 -1 0')):
            bodies[side]=ET.SubElement(tcp,'body',name=side)
            ET.SubElement(bodies[side],'joint',name=side+'_joint',type='slide',axis=axis,range='0 0.04')
        for k,row in enumerate(robot_cad):
            attrs={'name':f'robot_{k}','pos':values(row['position_m']),'quat':values(row['quaternion']),
                   'type':row['type'],'contype':'1','conaffinity':'6','margin':'0.01'}
            if row['type']=='mesh':
                name=f'mesh_{k}';ET.SubElement(asset,'mesh',name=name,vertex=values(row['vertices']),face=' '.join(str(x) for x in np.asarray(row['faces']).ravel()))
                attrs['mesh']=name
            else:attrs['size']=values(row['half_size'])
            ET.SubElement(bodies[row['body']],'geom',**attrs)
        centers,rotations,sizes=fixture_boxes(np.asarray(fixture_pose['position_m']),fixture_pose['yaw_rad'],fixture_cad,inflation)
        for i,(c,r,h) in enumerate(zip(centers,rotations,sizes)):
            ET.SubElement(world,'geom',name=f'fixture_{i}',type='box',pos=values(c),quat=values(mat_to_quat(r)),size=values(h),contype='2',conaffinity='1',margin='0.01')
        ET.SubElement(world,'geom',name='table',type='plane',pos=values([0,0,table_z]),size='2 2 .1',contype='2',conaffinity='1',margin='0.01')
        target=ET.SubElement(world,'body',name='observed_capsule',mocap='true')
        ET.SubElement(target,'geom',name='observed_capsule',type='capsule',size=values([capsule_cad['radius_m'],capsule_cad['half_length_m']]),contype='4',conaffinity='1',margin='0.01')
        self.xml=ET.tostring(root,encoding='unicode');self.model=mujoco.MjModel.from_xml_string(self.xml);self.data=mujoco.MjData(self.model)
        self.robot_ids=[];self.pad_sides={}
        for k,row in enumerate(robot_cad):
            gid=self.model.geom(f'robot_{k}').id;self.robot_ids.append(gid)
            if row['type']=='box' and row['body'] in ('left','right'):self.pad_sides[gid]=row['body']
        self.obstacle_ids={self.model.geom(f'fixture_{i}').id for i in range(5)}|{self.model.geom('table').id}
        self.capsule_id=self.model.geom('observed_capsule').id
        self.finger_qadr=[int(self.model.jnt_qposadr[self.model.joint(side+'_joint').id]) for side in ('left','right')]

    def set_capsule(self,position,axis):
        z=np.asarray(axis,float);z=z/np.linalg.norm(z);helper=np.eye(3)[np.argmin(np.abs(z))]
        x=np.cross(helper,z);x/=np.linalg.norm(x);y=np.cross(z,x)
        self.data.mocap_pos[0]=position;self.data.mocap_quat[0]=mat_to_quat(np.column_stack((x,y,z)))

    def update(self,position,rot,width):
        if not np.isfinite(np.r_[position,np.asarray(rot).ravel(),width]).all():raise ValueError('Nonfinite geometric query')
        self.data.qpos[:3]=position;self.data.qpos[3:7]=mat_to_quat(rot)
        self.data.qpos[self.finger_qadr]=width/2
        mujoco.mj_fwdPosition(self.model,self.data)

    def environment_clearance(self,position,rot,width):
        self.update(position,rot,width);minimum=.01
        for c in self.data.contact:
            if (c.geom1 in self.obstacle_ids and c.geom2 in self.robot_ids) or (c.geom2 in self.obstacle_ids and c.geom1 in self.robot_ids):
                minimum=min(minimum,float(c.dist))
        if not np.isfinite(minimum):raise ValueError('Nonfinite collision clearance')
        return minimum

    def capture(self,position,rot,opening,settings):
        """Static bilateral pad interception, not a force-closure/dynamics proof.

        Stop at the first width with opposed pad contacts of limited overlap.
        Target recentering, friction, compliance, slip and tracking are omitted.
        """
        best=None;cosine=np.cos(np.radians(settings['maximum_normal_angle_deg']))
        count=1+int(np.ceil(opening/settings['closing_step_m']))
        for width in np.linspace(opening,0,count):
            self.update(position,rot,width);sides={}
            for c in self.data.contact:
                if self.capsule_id not in (c.geom1,c.geom2):continue
                finger=int(c.geom2 if c.geom1==self.capsule_id else c.geom1)
                if finger not in self.pad_sides or not -settings['maximum_overlap_m']<=c.dist<=settings['touch_tolerance_m']:continue
                side=self.pad_sides[finger];normal=np.asarray(c.frame[:3])*(1 if finger==c.geom1 else -1)
                direction=rot[:,1]*(-1 if side=='left' else 1)
                alignment=float(normal@direction)
                if alignment>=cosine and (side not in sides or alignment>sides[side]['alignment']):
                    sides[side]={'alignment':alignment,'distance_m':float(c.dist),'position_m':np.asarray(c.pos).tolist()}
            if len(sides)==2:
                best={'joint_width_m':float(width),'pad_contacts':sides};break
        return {'geometrically_capturable':best is not None,'interception':best}
