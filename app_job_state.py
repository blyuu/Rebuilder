"""Pure job state contract, shared with the Qt process runner."""
TRANSITIONS={'queued':{'running','cancelled'},'running':{'completed','failed','cancelled'},
             'completed':set(),'failed':set(),'cancelled':set()}

def transition(record:dict,state:str):
    if state not in TRANSITIONS.get(record['status'],set()):
        raise ValueError(f'Invalid transition {record["status"]} -> {state}')
    record['status']=state
