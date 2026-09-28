const API_BASE = (window.AGROMIND_API_BASE || "").replace(/\/$/, "");
const form=document.getElementById("crop-form");
const result=document.getElementById("form-result");
form.addEventListener("submit",async(e)=>{
 e.preventDefault(); result.hidden=false; result.innerHTML="Running the research hybrid model…";
 result.style.background="#edf2e4";result.style.color="#38552b";
 const d=Object.fromEntries(new FormData(form).entries());
 for(const k of ["soil_ph","nitrogen","phosphorus","potassium","temperature_c","relative_humidity","water_requirement"])d[k]=Number(d[k]);
 try{
  const r=await fetch(`${API_BASE}/api/predict`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(d)});
  const p=await r.json();
  if(!r.ok)throw new Error(p.detail?.message||"Prediction service error");
  const recs=p.recommendations.map((x,i)=>`<li><strong>${i+1}. ${escapeHtml(x.crop)}</strong><span>Model score: ${(x.model_score*100).toFixed(2)}%</span></li>`).join("");
  const explain=p.explanation.map(x=>`<li>${escapeHtml(x.feature)} <small>(${(x.mask_importance*100).toFixed(1)}%)</small></li>`).join("");
  result.innerHTML=`<h3>Experimental crop suggestions — NOT FIELD-VALIDATED</h3><p class="result-warning">Prototype output. Several training features are not collected in this form and were set to zero. Scores are not calibrated farm-success probabilities.</p><ol class="prediction-list">${recs}</ol><h4>Model feature-mask signals (not causal agronomic explanations)</h4><ul>${explain}</ul><details><summary>Input and model limitations</summary><pre>${escapeHtml(JSON.stringify(p.input_mapping_warnings,null,2))}</pre></details><p>${escapeHtml(p.disclaimer)}</p>`;
 }catch(err){result.textContent=`Prediction unavailable: ${err.message}. Check that the API is running.`;result.style.background="#fff2d9";result.style.color="#7b4e13";}
});
function escapeHtml(s){return String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
