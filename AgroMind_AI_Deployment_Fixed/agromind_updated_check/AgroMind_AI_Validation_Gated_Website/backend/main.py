from pathlib import Path
import pickle
import json
import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from model_artifacts.model import Hybrid

ROOT=Path(__file__).resolve().parent

# Validation status is evidence-gated. This app must not call predictions validated
# until a documented, independently reviewed field-validation report is supplied.
VALIDATION_FILE=ROOT/"model_artifacts"/"validation_status.json"
def read_validation_status():
    default={
      "status":"research_prototype",
      "validated_for_recommendations":False,
      "scope":None,
      "validation_report":None,
      "reason":"No independently reviewed field-validation evidence is configured."
    }
    if not VALIDATION_FILE.exists():
        return default
    try:
        candidate=json.loads(VALIDATION_FILE.read_text(encoding="utf-8"))
        required=["status","validated_for_recommendations","scope","validation_report"]
        if not all(k in candidate for k in required):
            return default
        # The status file alone cannot prove validation; requires documented
        # approval fields and named validation report path.
        if candidate.get("validated_for_recommendations") is not True:
            return default | {"reason":"Validation evidence has not been approved."}
        if not candidate.get("validation_report") or not candidate.get("scope"):
            return default | {"reason":"Validation scope or report is missing."}
        return candidate
    except Exception:
        return default
ART=ROOT/"model_artifacts"
FEATURES=np.load(ART/"feature_columns.npy",allow_pickle=True).tolist()
with open(ART/"label_encoder.pkl","rb") as f:
    LABEL_ENCODER=pickle.load(f)
MODEL=Hybrid()
MODEL.load_state_dict(torch.load(ART/"best_reconstructed_hybrid.pth",map_location="cpu",weights_only=True))
MODEL.eval()
app=FastAPI(title="AgroMind AI Research Hybrid API",version="1.0.0")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_credentials=False,allow_methods=["GET","POST"],allow_headers=["*"])

class FarmerInputs(BaseModel):
    soil_ph: float=Field(ge=0,le=14)
    nitrogen: float=Field(ge=0)
    phosphorus: float=Field(ge=0)
    potassium: float=Field(ge=0)
    temperature_c: float
    relative_humidity: float=Field(ge=0,le=100)
    water_requirement: float=Field(ge=0)
    soil_type: str
    season: str
    water_source: str

def build_feature_vector(inp):
    # This maps the farmer form onto the dataset's 80-column training contract.
    # Features not collected by the form are zero-filled and explicitly flagged.
    d={c:0.0 for c in FEATURES}
    numeric={
      "SOIL_PH":inp.soil_ph,"TEMP":inp.temperature_c,"WATERREQUIRED":inp.water_requirement,
      "RELATIVE_HUMIDITY":inp.relative_humidity,"N":inp.nitrogen,"P":inp.phosphorus,"K":inp.potassium
    }
    for k,v in numeric.items():
        if k in d: d[k]=float(v)
    # normalize common soil/season/source values to training one-hot names
    soil_alias={
      "alluvial soil":"Alluvial soil","black soil":"Black Soil","clay soil":"Clay soil",
      "laterite soil":"Laterite soil","loamy soil":"Loamy soil","red soil":"Red soil","sandy soil":"Sandy soil"
    }
    soil=soil_alias.get(inp.soil_type.strip().lower(),inp.soil_type.strip())
    season=inp.season.strip().lower()
    water=inp.water_source.strip().lower()
    def set_one(prefix, value):
        candidates=[f"{prefix}_{value}",f"{prefix}_{value.lower()}",f"{prefix}_{value.title()}"]
        for c in candidates:
            if c in d: d[c]=1.0; return True
        # trim/case-insensitive match for data's inconsistent labels
        target=(prefix+"_"+value).strip().lower()
        matches=[c for c in d if c.lower().strip()==target]
        if matches: d[matches[0]]=1.0; return True
        return False
    soil_ok=set_one("SOIL",soil)
    season_ok=set_one("SEASON",season)
    water_ok=set_one("WATER_SOURCE",water)
    # Unknown/hidden training-time columns (high ranges, crop type, sow/harvest)
    # remain zero; predictions must be treated as provisional.
    return np.asarray([d[c] for c in FEATURES],dtype=np.float32),{
      "unprovided_training_features":["SOIL_PH_HIGH","CROPDURATION","CROPDURATION_MAX","MAX_TEMP","WATERREQUIRED_MAX","RELATIVE_HUMIDITY_MAX","N_MAX","P_MAX","K_MAX","TYPE_OF_CROP","SOWN","HARVESTED"],
      "categorical_match":{"soil":soil_ok,"season":season_ok,"water_source":water_ok}
    }

@app.get("/api/health")
def health():
    return {"status":"online","service":"AgroMind AI","model_status":"loaded","architecture":"reconstructed TabNet-inspired mask module + Transformer classifier","classes":len(LABEL_ENCODER.classes_),"feature_count":len(FEATURES),"validation":read_validation_status(),"warning":"Do not treat as validated agricultural advice unless independent field-validation status is approved for the applicable crop, region, season, and input range."}

@app.post("/api/predict")
def predict(inp:FarmerInputs):
    x,flags=build_feature_vector(inp)
    with torch.no_grad():
        logits,masks,_=MODEL(torch.from_numpy(x).unsqueeze(0))
        probs=torch.softmax(logits,dim=1)[0]
        vals,idx=torch.topk(probs,k=min(5,len(LABEL_ENCODER.classes_)))
    results=[]
    for p,i in zip(vals.tolist(),idx.tolist()):
        results.append({"crop":str(LABEL_ENCODER.inverse_transform([int(i)])[0]),"model_score":round(float(p),6)})
    importance=masks[0].mean(dim=0).numpy()
    top_indices=np.argsort(importance)[::-1][:8]
    explain=[{"feature":FEATURES[int(i)],"mask_importance":round(float(importance[i]),6)} for i in top_indices]
    return {"recommendations":results,"primary_recommendation":results[0]["crop"],"explanation":explain,"input_mapping_warnings":flags,"disclaimer":"Prototype prediction from reconstructed research architecture. Missing features were zero-filled; score is a model probability, not calibrated probability of farm success. Validate with local agricultural experts."}
