from flask import Flask, request, jsonify, send_from_directory
import cv2, numpy as np
app=Flask(__name__)

def measure(file):
    data=np.frombuffer(file.read(),np.uint8); img=cv2.imdecode(data,cv2.IMREAD_COLOR)
    if img is None: raise ValueError('Invalid image')
    h,w=img.shape[:2]; scale=min(1,1400/max(h,w)); img=cv2.resize(img,None,fx=scale,fy=scale) if scale<1 else img
    h,w=img.shape[:2]; gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5),0); edges=cv2.Canny(blur,45,140)
    cnts,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    quads=[]
    for c in cnts:
        p=cv2.arcLength(c,True); a=cv2.approxPolyDP(c,.025*p,True)
        if len(a)==4 and cv2.isContourConvex(a):
            area=cv2.contourArea(a)
            if .16*w*h<area<.98*w*h: quads.append((area,a.reshape(4,2).astype('float32')))
    if not quads: raise ValueError('Could not automatically find four card corners. Retake photo on a contrasting plain background with all edges visible.')
    pts=max(quads,key=lambda x:x[0])[1]; s=pts.sum(axis=1); d=np.diff(pts,axis=1).flatten()
    corners=np.array([pts[np.argmin(s)],pts[np.argmin(d)],pts[np.argmax(s)],pts[np.argmax(d)]],dtype=np.float32)
    dst=np.array([[0,0],[599,0],[599,839],[0,839]],dtype=np.float32)
    warp=cv2.warpPerspective(img,cv2.getPerspectiveTransform(corners,dst),(600,840))
    g=cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)
    # Look for the strongest near-continuous interior printed frame transition in each direction.
    # This is a heuristic and must never be represented as calibrated PSA grading.
    def boundary(profile,side):
        profile=np.convolve(profile,np.ones(7)/7,mode='same')
        dif=np.abs(np.gradient(profile.astype(float)))
        n=len(profile); lo=int(n*.025); hi=int(n*.22)
        if side=='end': return n-1-boundary(profile[::-1],'start')
        i=lo+int(np.argmax(dif[lo:hi])); strength=float(dif[i]);
        return i,strength
    # Ignore corners and use broad median strips to resist artwork and glare.
    px=np.median(g[150:690,:],axis=0); py=np.median(g[:,120:480],axis=1)
    l,sl=boundary(px,'start'); r,sr=boundary(px,'end'); t,st=boundary(py,'start'); b,sb=boundary(py,'end')
    left=float(l);right=float(599-r);top=float(t);bottom=float(839-b)
    def ratio(a,b): return [round(100*a/(a+b),1),round(100*b/(a+b),1)] if a+b>0 else None
    confidence='low' if min(sl,sr,st,sb)<3 or min(left,right,top,bottom)<5 else 'moderate'
    ok=(r>l and b>t and min(left,right,top,bottom)>=5)
    if not ok: confidence='low'
    import base64
    for x,y in [(l,0),(r,0)]: cv2.line(warp,(int(x),0),(int(x),839),(0,230,255),2)
    for x,y in [(0,t),(0,b)]: cv2.line(warp,(0,int(y)),(599,int(y)),(0,230,255),2)
    cv2.rectangle(warp,(0,0),(599,839),(70,240,100),3)
    _,buff=cv2.imencode('.jpg',warp,[cv2.IMWRITE_JPEG_QUALITY,85]); annotated='data:image/jpeg;base64,'+base64.b64encode(buff).decode()
    return {'horizontal':ratio(left,right),'vertical':ratio(top,bottom),'border_pixels':{'left':round(left,1),'right':round(right,1),'top':round(top,1),'bottom':round(bottom,1)},'confidence':confidence,'annotated':annotated,'warning':'Automated frame detection is heuristic. Verify yellow lines match the printed frame; full PSA grade cannot be determined reliably from these photos alone.'}

@app.get('/')
def home(): return send_from_directory('.','index.html')
@app.post('/api/measure')
def api():
    try:
        if 'front' not in request.files: return jsonify(error='Upload a front image'),400
        front=measure(request.files['front']); back=measure(request.files['back']) if request.files.get('back') else None
        return jsonify(front=front,back=back,grade=None,grade_status='Not determined: corner, edge, surface, print quality and eye appeal require validated inspection. No official PSA grade is generated.')
    except ValueError as e:return jsonify(error=str(e)),422
    except Exception:return jsonify(error='Image analysis failed'),500
if __name__=='__main__': app.run(host='127.0.0.1',port=5050,debug=False)
