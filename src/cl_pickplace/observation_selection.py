"""Optional checks across learned views using only observed geometry and priors."""
import numpy as np


def select_observation(views,target,settings,previous_fixture=None):
    candidates=[(camera,views[camera][target]) for camera in ('wrist_cam','top_cam')
                if views[camera][target]['accepted']]
    rejected={};floor=None
    if settings and target=='fixture' and len(candidates)==2:
        # The fixed-height horizontal fixture can have a rim mistaken for its
        # floor. Prefer the lower of two otherwise agreeing rectangles when
        # their vertical separation matches the known wall height.
        a,b=[c[1] for c in candidates]
        delta=abs(a['position_m'][2]-b['position_m'][2])
        xy=np.linalg.norm(np.asarray(a['position_m'])[:2]-np.asarray(b['position_m'])[:2])
        if (abs(delta-settings['nominal_wall_height_m'])<=settings['floor_alias_tolerance_m']
                and xy<=settings['fixture_xy_agreement_m']
                and abs(np.asarray(a['axis_world'])@np.asarray(b['axis_world']))>=np.cos(np.radians(10.))):
            candidates.sort(key=lambda c:c[1]['position_m'][2])
    if settings and target=='capsule':
        floors=[view['fixture']['position_m'][2] for view in views.values() if view['fixture']['accepted']]
        if floors:
            # A conservative lower envelope avoids excluding a genuine table
            # object merely because one view confused the rim and floor.
            floor=float(min(floors))
        elif previous_fixture is not None:
            floor=float(previous_fixture['position'][2])
        if floor is not None:
            good=[]
            for camera,candidate in candidates:
                if candidate['position_m'][2] < floor+settings['capsule_radius_m']-settings['floor_slack_m']:
                    rejected[camera]='capsule_below_observed_support_region'
                else:good.append((camera,candidate))
            candidates=good
        if len(candidates)==2:
            a,b=[c[1] for c in candidates]
            disagreement=(np.linalg.norm(np.asarray(a['position_m'])-b['position_m'])>settings['cross_view_position_m']
                          or abs(np.asarray(a['axis_world'])@np.asarray(b['axis_world']))<np.cos(np.radians(settings['cross_view_axis_deg'])))
            if disagreement:
                rejected.update({camera:'cross_view_disagreement' for camera,_ in candidates})
                candidates=[]
    chosen=None if not candidates else {**candidates[0][1],'source':'learned:'+candidates[0][0]}
    return chosen,{'rejected_views':rejected,'observed_fixture_floor_m':floor,
                   'cross_view_disagreement':any(v=='cross_view_disagreement' for v in rejected.values())}
