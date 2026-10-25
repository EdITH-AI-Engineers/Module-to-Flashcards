"""Browser interface, embedded so portable builds need no static-file lookup."""

from mistral_backend import (
    DEFAULT_MODEL, DEFAULT_MAX_RETRIES, DEFAULT_RATE_LIMIT_WAIT, DEFAULT_REQUEST_INTERVAL,
)

HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Module to Flashcards</title>
<style>
:root{font-family:system-ui,sans-serif;color:#17243b;background:#f2f5fa;font-size:15px}*{box-sizing:border-box}
body{margin:0}main{max-width:1100px;margin:48px auto;padding:0 24px}header{margin-bottom:28px}h1{font-size:32px;letter-spacing:-1px;margin:6px 0}h2{font-size:20px;margin:0 0 16px}p{line-height:1.6;color:#566378}.eyebrow{color:#2563eb;font-size:12px;font-weight:750;letter-spacing:2px}
.layout{display:grid;grid-template-columns:410px 1fr;gap:24px;align-items:start}.panel,.run{background:white;border:1px solid #dce3ed;border-radius:16px;padding:24px;box-shadow:0 6px 18px #17243b05}.panel{position:sticky;top:24px}
label{display:block;font-weight:650;margin-top:18px;margin-bottom:8px}input,select,button{font:inherit;border-radius:8px}input,select{padding:10px 12px;border:1px solid #cbd5e1;width:100%;background:white;color:#17243b}input:focus,select:focus{outline:2px solid #93b5ff;outline-offset:1px}input[type=file]{padding:12px;background:#f8faff}
button{cursor:pointer;padding:9px 12px;border:1px solid #cbd5e1;background:white;color:#253650;font-weight:600}button:hover{background:#edf3ff}button:disabled{opacity:.55;cursor:wait}.primary{width:100%;background:#2563eb;color:white;border-color:#2563eb;padding:12px;margin-top:24px}.primary:hover{background:#1d4ed8}.key{display:flex;gap:5px;margin:8px 0}.key input{min-width:0}.key button{padding:8px}.small{font-size:12px;line-height:1.6;color:#64748b}.row{display:flex;gap:12px}.row>div{flex:1}#runs{display:grid;gap:16px}.run h2{margin-bottom:6px}.run-head{display:flex;justify-content:space-between;align-items:center;gap:12px}.badge{font-size:12px;padding:5px 9px;border-radius:20px;background:#eef4ff;color:#2454a6}.module{border-top:1px solid #e5eaf2;padding-top:16px;margin-top:16px}.module strong{overflow-wrap:anywhere}.module p{font-size:13px;margin:7px 0}progress{width:100%;height:8px;accent-color:#2563eb}.error{color:#b42318;overflow-wrap:anywhere}a{color:#1d4ed8;display:inline-block;margin:8px 14px 0 0}.empty{text-align:center;padding:64px 25px;border:1px dashed #cbd5e1;border-radius:16px;color:#64748b}.keys-status{font-size:12px;color:#64748b;margin-top:14px;line-height:1.6}#notice{margin-top:14px;white-space:pre-wrap}.help{background:#edf4ff;border-radius:10px;padding:12px;font-size:12px;color:#34527d;margin-top:18px}
@media(max-width:820px){main{margin:24px auto}.layout{grid-template-columns:1fr}.panel{position:static}h1{font-size:28px}}
</style></head><body><main>
<header><div class="eyebrow">MISTRAL WORKSPACE</div><h1>Module to Flashcards</h1><p>Choose course folders of module PDFs. Tesseract reads scans; Mistral filters decoration and builds your flashcards.</p></header>
<div class="layout"><form class="panel" id="form"><h2>Start a new run</h2>
<label for="start-from">Start from</label><select id="start-from"><option value="modules">Module PDFs or text reports</option><option value="graphs">Saved graphs or structured modules</option></select>
<div id="graph-inputs" hidden><label for="saved-run">Saved run</label><select id="saved-run"><option value="">Choose a saved run</option></select><button type="button" id="refresh-saved" style="margin-top:8px">Refresh saved runs</button><div id="saved-selection" class="small" role="status"></div><div class="help">Continue from saved graphs or structured modules in your runs folder, including after a server restart. Structured modules skip PDF reading, OCR, and filtering, and continue with graph extraction. Available graphs and completed CSVs are reused. Your original run stays saved; results appear in a new run.</div></div>
<div id="module-inputs">
<label for="folder">Course folder</label><input id="folder" type="file" webkitdirectory directory multiple><div class="small">Choose one course folder, or a parent folder containing several courses. Each PDF is one module, e.g. CS0003 / Module 1.pdf.</div>
<label for="files">Or select individual modules</label><input id="files" type="file" accept=".pdf,application/pdf,.txt,text/plain" multiple><div id="selection" class="small" role="status"></div>
<label for="course">Course code for individual files</label><input id="course" placeholder="e.g. CS0003" autocomplete="off"><div class="small">Folder names supply course codes automatically. Name PDFs Module 1.pdf, M1.pdf, or CS0003-M1.pdf.</div>
<div class="row"><div><label for="ocr-language">OCR language</label><input id="ocr-language" value="eng" required aria-describedby="ocr-help"></div><div><label for="ocr-mode">OCR mode</label><select id="ocr-mode"><option value="auto">Automatic</option><option value="always">Every page</option></select></div></div>
<div id="ocr-help" class="small">Automatic OCR reads scanned pages and large embedded images. Language data must be installed for each selected language, e.g. eng or eng+fil.</div><div id="ocr-status" class="small" role="status">Checking Tesseract…</div>
</div>
<label>Mistral API key queue</label><div id="keys"></div><button type="button" id="add-key">+ Add backup key</button>
<div class="small">One key stays active. Temporary rate limits pause and retry that key. Backups are used only when a key is invalid or its quota is confirmed exhausted. Keys stay in server memory until you close the run or restart the server.</div>
<div class="row"><div><label for="model">Model</label><input id="model" value="__DEFAULT_MODEL__" required></div><div><label for="workers">Generation workers</label><select id="workers"><option selected>1</option><option>2</option><option>4</option><option>8</option><option>12</option><option>20</option></select></div></div>
<div class="small">The default model runs on Mistral's servers. Model access is checked before uploading modules. Requests pause for __REQUEST_INTERVAL__ seconds after each response. Temporary rate limits retry the active key up to __MAX_RETRIES__ attempts, waiting __RATE_LIMIT_WAIT__ seconds unless Mistral supplies a reset time. Extra workers wait; backup keys stay on standby.</div>
<div class="help">Each run has its own files, prompts, key queue, and results. Refreshing this page reconnects to runs on the same server. Mistral removes clearly decorative text while preserving lesson content, formulas, tables, and examples. Extracted text is sent to Mistral; OCR stays on your computer. Shared organization limits can still affect multiple keys.</div>
<button class="primary" id="start" type="submit">Generate flashcards</button><div id="notice" role="status" class="error"></div>
</form><section><div id="empty" class="empty">Your runs will appear here.<br><span class="small">Start another run while an earlier one is processing.</span></div><div id="runs" aria-live="polite"></div></section></div></main>
<script>
const $=id=>document.getElementById(id);
const watchedRuns=new Map();
const watchId=new URLSearchParams(location.search).get('watch');
const viewOnly=/^[a-f0-9]{32}$/.test(watchId||'');
let runNumber=0;
function selectedFiles(){return [...$('folder').files].filter(file=>/\.pdf$/i.test(file.name)).concat([...$('files').files]).sort((a,b)=>(a.webkitRelativePath||a.name).localeCompare(b.webkitRelativePath||b.name,undefined,{numeric:true}))}
function showSelection(){const files=selectedFiles();$('selection').textContent=files.length?`${files.length} module file(s) selected. `+files.slice(0,6).map(file=>file.webkitRelativePath||file.name).join(' · ')+(files.length>6?' …':''):''}
$('folder').onchange=showSelection;$('files').onchange=showSelection;
let savedRuns=[];
function showSavedSelection(){const run=savedRuns.find(run=>run.runId===$('saved-run').value);$('saved-selection').textContent=run?`${run.moduleCount} modules · ${run.graphCount} graphs · ${run.structuredCount} structured modules · ${run.completedCount} completed · ${run.pendingCount} remaining. Courses: ${run.courses.slice(0,6).join(', ')}${run.courses.length>6?' and '+(run.courses.length-6)+' more':''}.`+(run.errors.length?` ${run.errors.length} unusable saved module(s) will be reported.`:''):'Choose a run saved on this computer.'}
async function refreshSavedRuns(){const previous=$('saved-run').value;$('refresh-saved').disabled=true;$('saved-selection').textContent='Looking for saved graphs and structured modules…';try{const data=await jsonFetch('/api/saved-runs');savedRuns=data.runs;$('saved-run').replaceChildren(new Option(savedRuns.length?'Choose a saved run':'No saved modules found',''));for(const run of savedRuns){const label=`${new Date(run.updatedAt).toLocaleString()} · ${run.moduleCount} modules · ${run.pendingCount} remaining · ${run.runId.slice(0,8)}`+(run.running?' (still processing)':'');const option=new Option(label,run.runId);option.disabled=run.running||!run.moduleCount;$('saved-run').append(option)}if(savedRuns.some(run=>run.runId===previous&&!run.running))$('saved-run').value=previous;else if(savedRuns.filter(run=>!run.running&&run.moduleCount).length===1)$('saved-run').value=savedRuns.find(run=>!run.running&&run.moduleCount).runId;showSavedSelection();if(!savedRuns.length)$('saved-selection').textContent='No saved graphs or structured modules found in the runs folder.'}catch(error){$('saved-selection').textContent=error.message}finally{$('refresh-saved').disabled=false}}
$('refresh-saved').onclick=refreshSavedRuns;$('saved-run').onchange=showSavedSelection;
$('start-from').onchange=()=>{const graphs=$('start-from').value==='graphs';$('module-inputs').hidden=graphs;$('graph-inputs').hidden=!graphs;$('module-inputs').querySelectorAll('input,select').forEach(input=>input.disabled=graphs);$('start').textContent=graphs?'Continue to flashcards':'Generate flashcards';if(graphs)refreshSavedRuns()};
fetch('/api/ocr').then(response=>response.json()).then(data=>{$('ocr-status').textContent=data.message;if(!data.available)$('ocr-status').className='error small'}).catch(()=>{$('ocr-status').textContent='Could not check Tesseract. Check that the server is running.'});
function el(tag,text,cls){const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(cls)node.className=cls;return node}
function addKey(){const row=el('div',undefined,'key');const input=el('input');input.type='password';input.autocomplete='off';input.placeholder='Paste Mistral API key';input.setAttribute('aria-label','Mistral API key');row.append(input);
for(const [label,title,action] of [['↑','Move key earlier',()=>{if(row.previousElementSibling)row.parentNode.insertBefore(row,row.previousElementSibling)}],['↓','Move key later',()=>{if(row.nextElementSibling)row.parentNode.insertBefore(row.nextElementSibling,row)}],['×','Remove key',()=>{if($('keys').children.length>1)row.remove();else input.value=''}]]){const b=el('button',label);b.type='button';b.title=title;b.setAttribute('aria-label',title);b.onclick=action;row.append(b)}$('keys').append(row)}
$('add-key').onclick=addKey;addKey();
async function jsonFetch(url,options){const response=await fetch(url,options);const data=await response.json();if(!response.ok){const error=Error(typeof data.detail==='string'?data.detail:'Please check the API keys, model, and worker settings.');error.status=response.status;throw error}return data}
function renderRun(card,id,data){const detailsOpen=card.querySelector('details')?.open;card.replaceChildren();const waiting=data.keys.some(k=>k.state==='waiting');const head=el('div',undefined,'run-head');head.append(el('h2',card.dataset.title),el('span',data.running?(waiting?'Waiting for Mistral':'Processing'):data.summary.failed?'Finished with errors':'Complete','badge'));card.append(head);
if(data.model)card.append(el('p','Model: '+data.model,'small'));
if(data.active)card.append(el('p','Currently processing '+data.active.filename+': '+data.active.message,'small'));
card.append(el('div',data.keys.map(k=>`${k.label}: ${k.state}${k.retryAfterSeconds?' ('+k.retryAfterSeconds+'s)':''}`).join(' · '),'keys-status'));
const activeWait=data.keys.find(k=>k.state==='waiting');if(activeWait)card.append(el('p',`${activeWait.label} is waiting ${activeWait.retryAfterSeconds}s before retrying. Other keys stay on standby.`,'small'));
if(data.apiDiagnostics&&data.apiDiagnostics.length){const details=el('details');details.open=!!detailsOpen;details.append(el('summary','Mistral request details'));for(const entry of data.apiDiagnostics.slice(-3)){details.append(el('p',entry.message+(entry.requestId?' · Request '+entry.requestId:''),'small'))}card.append(details)}
for(const module of data.modules){const block=el('div',undefined,'module');block.append(el('strong',module.filename));const bar=el('progress');bar.max=100;bar.value=module.progressPercent;bar.setAttribute('aria-label',module.filename+' progress');block.append(el('p',module.message,module.error?'error':undefined),bar);
if(module.state==='completed'&&module.outputs.length===2){for(const part of [1,2]){const link=el('a','Download CSV '+part);link.href=`/api/sessions/${id}/files/${module.index}/${part}`;link.download='';block.append(link)}}card.append(block)}
if(!data.running&&!viewOnly){const close=el('button','Close run & clear keys');close.style.marginTop='16px';close.onclick=async()=>{try{await jsonFetch('/api/sessions/'+id,{method:'DELETE'});const run=watchedRuns.get(id);if(run){run.closed=true;clearTimeout(run.timer);watchedRuns.delete(id)}card.remove();if(!$('runs').children.length)$('empty').hidden=false}catch(error){$('notice').textContent=error.message}};card.append(close)}}

function watchRun(id,number,{awaitingStart=false}={}){
    if(watchedRuns.has(id))return watchedRuns.get(id);
    runNumber=Math.max(runNumber,number||0);
    const card=el('article',undefined,'run');card.dataset.title='Run '+(number||++runNumber);card.dataset.sessionId=id;
    card.append(el('h2',card.dataset.title),el('p',awaitingStart?'Starting…':'Reconnecting to saved progress…'));
    $('empty').hidden=true;$('runs').prepend(card);
    const run={id,card,awaitingStart,polling:false,closed:false,timer:null,refreshRequested:false};
    watchedRuns.set(id,run);refreshRun(run);return run;
}
function pollSoon(run){
    if(run.closed)return;
    clearTimeout(run.timer);
    if(run.polling){run.refreshRequested=true;return}
    refreshRun(run);
}
async function refreshRun(run){
    if(run.closed||run.polling)return;
    run.polling=true;let keepPolling=true;
    try{
        const status=await jsonFetch('/api/sessions/'+run.id);
        if(run.closed)return;
        if(status.runNumber){runNumber=Math.max(runNumber,status.runNumber);run.card.dataset.title='Run '+status.runNumber}
        if(status.modules.length){run.awaitingStart=false;renderRun(run.card,run.id,status)}
        else if(!run.awaitingStart){run.card.replaceChildren(el('h2',run.card.dataset.title),el('p','No modules were started in this run.','small'))}
        keepPolling=status.running||run.awaitingStart;
    }catch(error){
        if(run.closed)return;
        keepPolling=error.status!==404;
        run.card.querySelector('.connection-error')?.remove();
        run.card.append(el('p',error.status===404?'This run is no longer available. The server may have restarted.':'Connection interrupted. Reconnecting to this run…','error connection-error'));
    }finally{
        run.polling=false;
        if(!run.closed&&(keepPolling||run.refreshRequested)){
            const delay=run.refreshRequested?0:1200;run.refreshRequested=false;
            run.timer=setTimeout(()=>refreshRun(run),delay);
        }
    }
}
let restoring=false;let restoreTimer;
async function restoreRuns(){
    if(restoring||viewOnly)return;
    restoring=true;clearTimeout(restoreTimer);
    try{
        const data=await jsonFetch('/api/sessions');
        for(const session of data.sessions)watchRun(session.sessionId,session.runNumber);
        if($('notice').dataset.reconnecting){$('notice').textContent='';delete $('notice').dataset.reconnecting}
    }catch(error){
        $('notice').textContent='Could not reconnect to saved runs. Check that the server is running.';
        $('notice').dataset.reconnecting='true';
        if(error.status!==404)restoreTimer=setTimeout(restoreRuns,5000);
    }finally{restoring=false}
}
if(viewOnly){
    $('form').hidden=true;$('form').style.display='none';document.querySelector('.layout').style.gridTemplateColumns='1fr';
    document.querySelector('header p').textContent='Live progress of your existing run. Refreshing this view reconnects without restarting processing.';
    watchRun(watchId,1);
}else restoreRuns();
function reconnectRuns(){restoreRuns();for(const run of watchedRuns.values())pollSoon(run)}
window.addEventListener('pageshow',reconnectRuns);
document.addEventListener('visibilitychange',()=>{if(!document.hidden)reconnectRuns()});
$('form').onsubmit=async event=>{event.preventDefault();$('notice').textContent='';$('notice').className='error';const keys=[...$('keys').querySelectorAll('input')].map(i=>i.value.trim()).filter(Boolean);if(!keys.length){$('notice').textContent='Enter at least one Mistral API key.';return}const fromGraphs=$('start-from').value==='graphs';const sourceRunId=$('saved-run').value;const files=selectedFiles();if(fromGraphs&&!sourceRunId){$('notice').textContent='Choose a saved run with graphs or structured modules.';return}if(!fromGraphs&&!files.length){$('notice').textContent='Choose a course folder containing module PDFs, or select individual files.';return}const body=new FormData();body.append('courseCode',$('course').value);files.forEach(file=>body.append('files',file,file.webkitRelativePath||file.name));$('start').disabled=true;let id;let processingStarted=false;
try{const session=await jsonFetch('/api/sessions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({apiKeys:keys,model:$('model').value.trim(),workers:Number($('workers').value),ocrLanguage:fromGraphs?'eng':$('ocr-language').value.trim(),ocrMode:fromGraphs?'auto':$('ocr-mode').value})});id=session.sessionId;
$('notice').className='small';$('notice').textContent='Checking model access with Mistral…';await jsonFetch('/api/sessions/'+id+'/check',{method:'POST'});$('notice').textContent='';$('notice').className='error';
$('keys').querySelectorAll('input').forEach(i=>i.value='');const run=watchRun(id,session.runNumber,{awaitingStart:true});
processingStarted=true;const processing=fromGraphs?jsonFetch('/api/sessions/'+id+'/resume',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sourceRunId})}):jsonFetch('/process?session_id='+encodeURIComponent(id),{method:'POST',body});$('start').disabled=false;if(!fromGraphs){$('files').value='';$('folder').value='';showSelection()}processing.then(()=>{run.awaitingStart=false;pollSoon(run);if(fromGraphs)refreshSavedRuns()}).catch(error=>{run.awaitingStart=false;pollSoon(run);$('notice').textContent=error.message});
}catch(error){$('notice').className='error';$('notice').textContent=error.message;if(id&&!processingStarted){try{await jsonFetch('/api/sessions/'+id,{method:'DELETE'})}catch{}}}finally{$('start').disabled=false}};
</script></body></html>'''.replace('__DEFAULT_MODEL__', DEFAULT_MODEL).replace(
    '__REQUEST_INTERVAL__', f'{DEFAULT_REQUEST_INTERVAL:g}').replace(
    '__MAX_RETRIES__', str(DEFAULT_MAX_RETRIES)).replace(
    '__RATE_LIMIT_WAIT__', f'{DEFAULT_RATE_LIMIT_WAIT:g}')
