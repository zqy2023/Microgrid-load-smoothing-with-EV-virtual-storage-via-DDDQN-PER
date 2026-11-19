"""Bound model-ensemble and offline action-score sources. Training only."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

from evaluate_single import ROOT, native


class EnsembleGroup:
    """Batched inference of the sealed advantage streams."""
    def __init__(self, states):
        def stack(name):
            return torch.stack([s[name] for s in states])
        self.layers = [
            (stack(n+".weight"),stack(n+".bias")) for n in (
                "feature_layers.0","feature_layers.3",
                "advantage_stream.0","advantage_stream.2",
            )
        ]
        self.size = len(states)

    def forward(self, obs):
        hidden=obs.unsqueeze(0).expand(self.size,-1,-1)
        for i,(weight,bias) in enumerate(self.layers):
            hidden=torch.bmm(hidden,weight.transpose(1,2))+bias[:,None,:]
            if i<3:
                hidden=hidden.relu()
        return (hidden-hidden.mean(-1,keepdim=True)).mean(0)


class TeachingTargets:
    def __init__(self, data, use_ensemble):
        self.data=data
        self.base=None
        self.cost=None
        self.dp=None
        self.bindings={}
        self.counts={"1":dict(batches=0,rows=0,labels=0),
                     "2":dict(batches=0,rows=0,labels=0)}
        if use_ensemble:
            receipt=ROOT/"config"/"ensemble_teacher.json"
            self.bindings["ensemble_receipt"]=native.file_binding(receipt)
            rec=json.loads(receipt.read_text(encoding="utf8"))
            assert rec["weight"]==.1
            assert {k:v["sha256"] for k,v in rec["inputs"].items()}==data.hashes
            groups={"base":[],"cost":[]}
            self.bindings["ensemble_models"]=rec["members"]
            for member in rec["members"]:
                path=native.resolve(member["path"])
                assert native.sha(path)==member["sha256"]
                model=torch.load(path,map_location="cpu",weights_only=True)
                state=model["q_network_state_dict"] if member["kind"]=="archived" else model
                groups[member["role"]].append(state)
            assert len(groups["base"])==8 and len(groups["cost"])==2
            self.base=EnsembleGroup(groups["base"])
            self.cost=EnsembleGroup(groups["cost"])

    def attach_dp(self, dp_manifest):
        sys.path.insert(0,str(ROOT/"src"))
        from offline_teacher import Teacher
        self.dp=Teacher(Path(dp_manifest))
        self.bindings["dp_manifest"]=native.file_binding(dp_manifest)
        self.bindings["dp_source"]=native.file_binding(sys.modules[Teacher.__module__].__file__)

    @staticmethod
    def normalize(score):
        return score/score.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6)

    @torch.no_grad()
    def scores(self, stage, obs, source, energy):
        available=self.data.ev[source]>.1
        count=self.counts[str(stage)]
        count["batches"]+=1
        count["rows"]+=len(source)
        count["labels"]+=int(available.sum())
        if stage==1:
            score=.9*self.normalize(self.base.forward(obs))+.1*self.normalize(self.cost.forward(obs))
        elif stage==2:
            scores=self.dp.action_values(source,energy)
            score=torch.from_numpy((scores-scores.mean(axis=1,keepdims=True)).astype(np.float32))
        else:
            raise ValueError("No teacher for stage zero")
        return score,available

