// [COMBO] {"material":"ui_editor_properties_blend_mode","combo":"BLENDMODE","type":"imageblending","default":1}

#define PI 3.1415927
#define STEPS 100
#define INSIDE_STEPS 20
#define AO_STEPS 8
#define SMOOTHING_VAL 0.06
#define numBlobs 80

#define fract frac

uniform vec3 u_LiquidColor; // {"group":"液体核心","material":"液体颜色","type":"color","default":"0.0,0.0,0.0"}
uniform float u_LiquidAlpha; // {"group":"液体核心","material":"透明度","default":1.0,"range":[0.0,1.0]}
uniform float u_CenterConcentration; // {"group":"液体核心","material":"中心浓度","default":1.0,"range":[0.1,3.0]}
uniform float u_LiquidFusion; // {"group":"液体核心","material":"液体融合度","default":1.0,"range":[0.1,2.0]}
uniform float u_EdgeSoftness; // {"group":"液体核心","material":"边缘柔和","default":1.0,"range":[0.1,2.0]}
uniform float u_EdgeSmoothness; // {"group":"液体核心","material":"抗锯齿","default":0.05,"range":[0.001,1]}
uniform float u_MinBlobSize; // {"group":"液体核心","material":"球体最小尺寸限制","default":0.02,"range":[0.0,0.3]}

uniform float u_AnimationSpeed; // {"group":"动画控制","material":"动画速度","default":1.0,"range":[0.1,3.0]}
uniform float u_SphereCount; // {"group":"动画控制","material":"球体数量","default":80.0,"range":[20.0,120.0]}
uniform float u_LiquidSize; // {"group":"动画控制","material":"液体大小","default":1.0,"range":[0.5,2.0]}
uniform float u_MotionRange; // {"group":"动画控制","material":"运动范围","default":1.0,"range":[0.0,3.0]}

uniform vec3 u_EdgeColor1; // {"group":"边缘光效","material":"边缘颜色1","type":"color","default":"0.2,0.5,0.9"}
uniform vec3 u_EdgeColor2; // {"group":"边缘光效","material":"边缘颜色2","type":"color","default":"0.871,0.435,0.086"}
uniform float u_EdgeIntensity; // {"group":"边缘光效","material":"边缘光强度","default":1.0,"range":[0.0,2.0]}
uniform float u_EdgeWidth; // {"group":"边缘光效","material":"边缘光宽度","default":1.0,"range":[0.1,3.0]}
uniform float u_GradientRotation; // {"group":"边缘光效","material":"渐变旋转角度","default":0.0,"range":[0.0,360.0]}

uniform float u_CameraDistance; // {"group":"变换控制","material":"镜头距离","default":4.5,"range":[2.0,10.0]}
uniform float u_ScaleDistance; // {"group":"变换控制","material":"缩放距离","default":1.0,"range":[0.5,3.0]}
uniform float u_AngleX; // {"group":"变换控制","material":"X轴角度","default":0.0,"range":[-180.0,180.0]}
uniform float u_AngleY; // {"group":"变换控制","material":"Y轴角度","default":0.0,"range":[-180.0,180.0]}

uniform vec3 u_HighlightColor1; // {"group":"高光区域 1","material":"高光颜色 1","type":"color","default":"1.0,1.0,1.0"}
uniform float u_HighlightIntensity1; // {"group":"高光区域 1","material":"高光强度 1","default":0.5,"range":[0.0,2.0]}
uniform float u_HighlightConcentration1; // {"group":"高光区域 1","material":"高光凝聚度 1","default":16.0,"range":[1.0,128.0]}
uniform float u_HorizontalAngle1; // {"group":"高光区域 1","material":"水平角度 1","default":136,"range":[0.0,360.0]}
uniform float u_VerticalAngle1; // {"group":"高光区域 1","material":"垂直角度 1","default":180,"range":[0.0,360.0]}

uniform vec3 u_HighlightColor2; // {"group":"高光区域 2","material":"高光颜色 2","type":"color","default":"0.8,0.8,1.0"}
uniform float u_HighlightIntensity2; // {"group":"高光区域 2","material":"高光强度 2","default":0.3,"range":[0.0,2.0]}
uniform float u_HighlightConcentration2; // {"group":"高光区域 2","material":"高光凝聚度 2","default":16.0,"range":[1.0,128.0]}
uniform float u_HorizontalAngle2; // {"group":"高光区域 2","material":"水平角度 2","default":60,"range":[0.0,360.0]}
uniform float u_VerticalAngle2; // {"group":"高光区域 2","material":"垂直角度 2","default":20,"range":[0.0,360.0]}

uniform vec2 g_TexelSize;
uniform float g_Time;

#include "common.h"

varying vec4 v_TexCoord;

const float fl = 2.0;
const vec3 lookAt = vec3(0.0, 0.0, 0.0);


vec4 hash41(float src) {
    vec4 p4 = fract(vec4(src, src, src, src) * vec4(0.1031, 0.1136, 0.1375, 0.1543));
    p4 += dot(p4, p4.wzxy + 33.33);
    return fract((p4.xxyz + p4.yzzw) * p4.zywx);
}

float smin(float a, float b, float k) {
    k *= 6.0;
    float h = max(k - abs(a - b), 0.0) / k;
    return min(a, b) - h * h * h * k * (1.0 / 6.0);
}

vec4 GetBlob(int i, float time) {
    vec4 rand1 = hash41(float(i));
    vec4 rand2 = hash41(float(i) * 1.3145);
    
    vec2 minmaxFreq = vec2(0.4, 2.8);
    vec2 minmaxPhaseOffset = vec2(0.0, 2.0 * PI);
    vec2 minmaxRadius = vec2(0.15, 0.5);
    vec2 minmaxMoveRadius = vec2(0.2, 1.0);
    
    vec3 freq = mix(minmaxFreq.xxx, minmaxFreq.yyy, rand1.xyz);
    vec3 phase = mix(minmaxPhaseOffset.xxx, minmaxPhaseOffset.yyy, rand2.xyz);
    float moveRad = mix(minmaxMoveRadius.x, minmaxMoveRadius.y, pow(rand2.w, 1.0));
    float rad = mix(minmaxRadius.x, minmaxRadius.y, pow(rand1.w, 2.0)) * exp(-moveRad * 1.75);
    rad = max(rad, u_MinBlobSize);
    
    freq *= u_AnimationSpeed;
    rad *= u_LiquidSize;
    moveRad *= u_MotionRange;
    vec3 bp_chaos = vec3(
        sin(time * freq.x + phase.x), 
        cos(time * freq.y + phase.y),
        sin(time * freq.z + phase.z)) * vec3(moveRad, moveRad, moveRad);
    vec3 bp = bp_chaos;
    
    return vec4(bp, rad);
}

float map(vec3 p) {
    float d = 100000.0;
    int blobCount = int(u_SphereCount);
    
    for (int i = 0; i < blobCount; i++) {
        if (i >= numBlobs) break;
        
        vec4 blob = GetBlob(i, g_Time);
        vec3 blobPos = blob.xyz;
        float rad = blob.w;
        
        float blobDist = length(p - blobPos) - rad;
        d = smin(d, blobDist, SMOOTHING_VAL * u_LiquidFusion);
    }
    return d;
}

float March(vec3 ro, vec3 rd, out float endD, out int stepsTaken) {
    float t = 0.0;
    for (stepsTaken = 0; stepsTaken < STEPS; stepsTaken++) {
        vec3 p = ro + t * rd;
        float d = map(p);
        endD = d;
        if (d < 0.001) return t;
        t += d;
        if (t > 20.0) return t;
    }
    return t;
}

float InsideMarch(vec3 ro, vec3 rd) {
    float t = 0.0;
    float accumD = 0.0;
    for (int i = 0; i < INSIDE_STEPS; i++) {
        vec3 p = ro + t * rd;
        float d = map(p);
        if (d < 0.0) {
            accumD += -d;
        }
        t += 0.05;
    }
    return accumD;
}

vec3 Normal(vec3 p) {
    const float h = 0.001;
    const vec2 k = vec2(1.0, -1.0);
    return normalize(k.xyy * map(p + k.xyy * h) + 
                     k.yyx * map(p + k.yyx * h) + 
                     k.yxy * map(p + k.yxy * h) + 
                     k.xxx * map(p + k.xxx * h));
}

float AO(vec3 pos, vec3 nor) {
    float occ = 0.0;
    float sca = 1.0;
    for (int i = 0; i < AO_STEPS; i++) {
        float t = 0.01 + 0.08 * float(i);
        float d = map(pos + t * nor);
        occ += (t - d) * sca;
        sca *= 0.85;
    }
    return clamp(1.0 - occ / 3.14, 0.0, 1.0);
}

vec3 Render(vec3 ro, vec3 rd, float d) {
    vec3 p = ro + d * rd;
    vec3 nor = Normal(p);
    float thickness = InsideMarch(p - nor * 0.01, rd);
    
    vec3 lightDir = normalize(vec3(-1.0, 2.0, 0.0));
    vec3 refl = reflect(rd, nor);
    vec3 refr = refract(rd, nor, 1.0 / 1.4);
    if (length(refr) == 0.0) refr = refl;
    
    float fresnel = abs(dot(rd, nor));
    float lightDot = dot(nor, lightDir);
    float ssFake = saturate(lightDot * 0.5 + 0.5);
    float spec = saturate(dot(refl, lightDir));
    spec = pow(spec, 256.0);
    float ao = AO(p, nor);
    
    vec3 transformedP = p;
    
    
    vec3 liquidCol = u_LiquidColor;
    
    float ambientStrength = 0.2 + 0.6 * clamp((u_CenterConcentration - 0.1) / 2.9, 0.0, 1.0);
    
    
    float aoFactor = ao * 0.5 + 0.5;
    
    vec3 lighting = mix(vec3(ambientStrength, ambientStrength, ambientStrength), vec3(1.0, 1.0, 1.0), ssFake);
    
    liquidCol *= lighting * aoFactor;
    
    vec3 highlight1Dir = normalize(vec3(
        cos(u_HorizontalAngle1 * PI / 180.0) * sin(u_VerticalAngle1 * PI / 180.0),
        cos(u_VerticalAngle1 * PI / 180.0),
        sin(u_HorizontalAngle1 * PI / 180.0) * sin(u_VerticalAngle1 * PI / 180.0)
    ));
    vec3 highlight2Dir = normalize(vec3(
        cos(u_HorizontalAngle2 * PI / 180.0) * sin(u_VerticalAngle2 * PI / 180.0),
        cos(u_VerticalAngle2 * PI / 180.0),
        sin(u_HorizontalAngle2 * PI / 180.0) * sin(u_VerticalAngle2 * PI / 180.0)
    ));
    
    vec3 refDir = reflect(rd, nor);
    float highlight1 = saturate(dot(refDir, highlight1Dir));
    float highlight2 = saturate(dot(refDir, highlight2Dir));
    
    highlight1 = pow(highlight1, u_HighlightConcentration1);
    highlight2 = pow(highlight2, u_HighlightConcentration2);
    
    liquidCol += u_HighlightColor1 * highlight1 * u_HighlightIntensity1;
    liquidCol += u_HighlightColor2 * highlight2 * u_HighlightIntensity2;
    
    float gradientAngle = u_GradientRotation * PI / 180.0;
    vec2 rotDir = vec2(cos(gradientAngle), sin(gradientAngle));
    float gradientPos = dot(transformedP.xy, rotDir);
    float gradientFactor = gradientPos * 0.5 + 0.5;
    gradientFactor = clamp(gradientFactor, 0.0, 1.0);
    
    vec3 edgeLightColor = mix(u_EdgeColor1, u_EdgeColor2, gradientFactor);
    
    float rimFactor = 1.0 - abs(dot(rd, nor));
    
    float rimPower = 4.0 / max(0.1, u_EdgeWidth);
    float rimIntensity = pow(rimFactor, rimPower);
    
    rimIntensity *= u_EdgeIntensity;
    
    liquidCol += edgeLightColor * rimIntensity;
    
    vec3 finalCol = liquidCol;
    
    
    return clamp(finalCol, vec3(0.0, 0.0, 0.0), vec3(1.0, 1.0, 1.0));
}

void main() {
    vec2 uv = v_TexCoord.xy * 2.0 - 1.0;
    
    uv.x *= g_TexelSize.y / g_TexelSize.x;
    
    vec3 ro = vec3(0.0, 0.0, u_CameraDistance);
    vec3 cf = normalize(lookAt - ro);
    vec3 cr = normalize(cross(cf, vec3(0.0, 1.0, 0.0)));
    vec3 cu = normalize(cross(cr, cf));

    float angleXRad = u_AngleX * PI / 180.0;
    float angleYRad = u_AngleY * PI / 180.0;
    
    mat3 rotX = mat3(
        vec3(1.0, 0.0, 0.0),
        vec3(0.0, cos(angleXRad), -sin(angleXRad)),
        vec3(0.0, sin(angleXRad), cos(angleXRad))
    );
    
    mat3 rotY = mat3(
        vec3(cos(angleYRad), 0.0, sin(angleYRad)),
        vec3(0.0, 1.0, 0.0),
        vec3(-sin(angleYRad), 0.0, cos(angleYRad))
    );
    
    vec3 newRo = ro;
    newRo = mul(newRo, rotX);
    newRo = mul(newRo, rotY);
    
    vec3 newCf = normalize(lookAt - newRo);
    vec3 newCr = normalize(cross(newCf, vec3(0.0, 1.0, 0.0)));
    vec3 newCu = normalize(cross(newCr, newCf));
    
    vec3 rd = normalize(uv.x * newCr + uv.y * newCu + fl * newCf);
    
    float endD;
    int stepsTaken;
    float d = March(newRo, rd, endD, stepsTaken);
    
    float aaThreshold = u_EdgeSmoothness;
    float edgeAlpha = 1.0 - smoothstep(0.001, 0.001 + aaThreshold, endD);
    
    vec3 col = vec3(0.0, 0.0, 0.0);
    
    if (edgeAlpha > 0.0) {
        col = Render(newRo, rd, d);
    }
    
    float alpha = u_LiquidAlpha * edgeAlpha;
    
    col = pow(col, vec3(1.0 / 2.2, 1.0 / 2.2, 1.0 / 2.2));
    
    gl_FragColor = vec4(col, alpha);
}
