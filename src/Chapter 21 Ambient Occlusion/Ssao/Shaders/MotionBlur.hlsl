#define _ALLOW_COMPILER_AND_LINKER_FEEDBACK_

#ifndef BlockSizeX
#define BlockSizeX 8
#endif

#ifndef BlockSizeY
#define BlockSizeY 8
#endif

#ifndef NMRadius
#define NMRadius 2
#endif

#ifndef TileSizeDebug
#define TileSizeDebug 8
#endif

struct VS_OUT
{
    float4 PosH : SV_POSITION;
    float2 uv : TEXCOORD0;
};

VS_OUT VS(uint vid : SV_VertexID)
{
    float2 positions[3] =
    {
        float2(-1.0f, -1.0f),
        float2(-1.0f, 3.0f),
        float2(3.0f, -1.0f)
    };

    VS_OUT o;
    float2 pos = positions[vid];

    o.PosH = float4(pos, 0.0f, 1.0f);
    o.uv = pos * 0.5f + 0.5f;

    return o;
}

cbuffer MatricesCb : register(b0)
{
    float4x4 InverseProjMatrix_;
    float4x4 viewMatrixInvCur_;
    float4x4 viewMatrixInvPrev_;
    float4x4 ReprojectionMatrix_;
    
    int2 gTexSizeV; // {width, height}
    float2 paddingV;
    
    float4 gNearFar;
   // float2 padding2;
};

Texture2D<float> currentDepth : register(t0);
Texture2D<float> previousDepth : register(t1);
SamplerState samLinear : register(s0);
SamplerState samPoint : register(s1);

float3 WorldPosFromDepth(float depth, float2 uv, row_major float4x4 projInv, row_major float4x4 viewInv)
{
    // depth stored in [0,1] NDC z; convert to clip z, reconstruct view/world.
    float z = depth * 2.0f - 1.0f;
    float4 clip = float4(uv * 2.0f - 1.0f, z, 1.0f);

    float4 view = mul(projInv, clip);
    view /= view.w;

    float4 world = mul(viewInv, view);
    return world.xyz;
}

float2 PSMain(VS_OUT input) : SV_Target
{
    float2 uv = input.uv;

    float curZ = currentDepth.Sample(samLinear, uv).r;
    float prevZ = previousDepth.Sample(samLinear, uv).r;

    float3 worldPrev = WorldPosFromDepth(prevZ, uv, InverseProjMatrix_, viewMatrixInvPrev_);
    float3 worldCur = WorldPosFromDepth(curZ, uv, InverseProjMatrix_, viewMatrixInvCur_);

    float2 velocity = worldCur.xy - worldPrev.xy;

    return velocity;
}

RWTexture2D<float2> VelocityOut : register(u2);

float4 GetCurrentNDC(int2 pix, float depth)
{
    float2 uv = (float2(pix) + 0.5f) * (1.0 / float2(gTexSizeV));
    float z = depth * 2.0f - 1.0f;
    return float4(uv * 2.0f - 1.0f, z, 1.0f);
}

// Reconstruct world position from depth sampled at integer pixel coords.
// We use Load so coordinates are integers (x,y,0); depth texel value is in [0..1].
float3 WorldPosFromDepthAtPixel(float depth, int2 px)
{
    float2 uv = (float2(px) + 0.5f) / (float2) gTexSizeV; // center of pixel
    float z = depth * 2.0f - 1.0f;
    float4 clip = float4(uv * 2.0f - 1.0f, z, 1.0f);
    float4 view = mul(InverseProjMatrix_, clip);
    view /= view.w;
    float4 world = mul(viewMatrixInvCur_, view);
    return world.xyz;
}

// If previous view matrix differs, we need a different function:
float3 WorldPosFromDepthAtPixelPrev(float depth, int2 px)
{
    float2 uv = (float2(px) + 0.5f) / (float2) gTexSizeV;
    float z = depth * 2.0f - 1.0f;
    float4 clip = float4(uv * 2.0f - 1.0f, z, 1.0f);
    float4 view = mul(InverseProjMatrix_, clip);
    view /= view.w;
    float4 world = mul(viewMatrixInvPrev_, view);
    return world.xyz;
}

[numthreads(BlockSizeX, BlockSizeY, 1)]
void CSMainWorld(uint3 DTid : SV_DispatchThreadID)
{
    int2 pix = int2(DTid.xy);

    // bounds check (some threads out-of-range on edge)
    if (pix.x >= gTexSizeV.x || pix.y >= gTexSizeV.y)
        return;

    float curZ = currentDepth.Load(int3(pix, 0));
    float prevZ = previousDepth.Load(int3(pix, 0));

    float3 worldCur = WorldPosFromDepthAtPixel(curZ, pix);
    float3 worldPrev = WorldPosFromDepthAtPixelPrev(prevZ, pix);

    float2 velocity = worldCur.xy - worldPrev.xy;

    // optionally scale to screen-space pixels if you want (depends on later passes)
    VelocityOut[pix] = velocity;
}

float LinearizeDepthV(float depth)
{
    float gNear = gNearFar.x;
    float gFar = gNearFar.y;
    return (gNear * gFar) / (gFar - depth * (gFar - gNear));
}

[numthreads(BlockSizeX, BlockSizeY, 1)]
void CSMain(uint3 DTid : SV_DispatchThreadID)
{
    int2 pix = int2(DTid.xy);

    // bounds check
    if (pix.x >= gTexSizeV.x || pix.y >= gTexSizeV.y)
    {
        VelocityOut[pix] = float2(0, 0);
        return;
    }

        // --- читаем depth ---
    float depth = currentDepth.Load(int3(pix, 0)).r;
    depth = LinearizeDepthV(depth);

    // --- 1) UV и clip space ---
    float2 invSize = 1.0 / float2(gTexSizeV);
    float2 uv = (float2(pix) + 0.5f) * invSize;
    float z = depth; //depth * 2.0f - 1.0f;

    // сначала восстанавливаем view-space / clip-space через inverse projection
    float4 clip = float4(uv * 2 - 1, z, 1);
    float4 viewPos = mul(InverseProjMatrix_, clip);
    viewPos /= viewPos.w;

    // --- 2) Репроекция в прошлый кадр ---
    float4 prevClip = mul(ReprojectionMatrix_, viewPos);
    prevClip /= prevClip.w;

    // --- 3) Переводим в UV с Y переворотом (DirectX) ---
    float2 currentVP = viewPos.xy * float2(0.5f, -0.5f) + 0.5f;
    float2 prevVP = prevClip.xy * float2(0.5f, -0.5f) + 0.5f;

    // --- 4) Velocity в пикселях ---
    float2 velocity = (currentVP - prevVP) * gTexSizeV;

    VelocityOut[pix] = velocity;

}


////////////////////////////////////////////////////////

// Dynamic texture size provided by the app
cbuffer TileMaxCB : register(b1)
{
    float2 gTexSize; // screen resolution
    float2 gInvTexSize; // 1 / resolution
    int TileSize; // usually 8
    float padding;
};

Texture2D<float2> velocityBuffer : register(t2);

RWTexture2D<float2> tileMaxOut : register(u0);

//groupshared float3 sharedVel[BlockSizeX * BlockSizeY];

/*
[numthreads(BlockSizeX, BlockSizeY, 1)]
void CS_TileMax(uint3 groupId : SV_GroupID,
                uint3 groupThreadId : SV_GroupThreadID)
{
    //groupshared float3 sharedVel[BlockSizeX * BlockSizeY];
    // tile index in tile grid:
    uint2 tileID = groupId.xy;

    // pixel offset inside this tile:
    uint2 localID = groupThreadId.xy;

    uint2 pixel = tileID * TileSize + localID;

    //groupshared float3 sharedVel[BlockSizeX * BlockSizeY];

    uint idx = localID.y * TileSize + localID.x;
    //uint idx = localID.y * BlockSizeX + localID.x;

    float2 vel = float2(0, 0);

    // check bounds (edges of screen)
    if (pixel.x < gTexSize.x && pixel.y < gTexSize.y)
    {
        vel = velocityBuffer[pixel];
    }

    // store squared length so we don't recompute it later
    sharedVel[idx] = float3(vel, dot(vel, vel));

    GroupMemoryBarrierWithGroupSync();

    // reduction: only thread (0,0) finds max
    if (idx == 0)
    {
        float3 best = float3(0, 0, 0); //sharedVel[0];

        const uint total = BlockSizeX * BlockSizeY;

        [unroll]
        for (uint i = 1; i < total; i++)
        {
            if (sharedVel[i].z > best.z)
                best = sharedVel[i];
        }

        tileMaxOut[tileID] = best.xy;
    }
}*/

/*[numthreads(BlockSizeX, BlockSizeY, 1)]
void CS_TileMax(uint3 DTid : SV_DispatchThreadID)
{
    uint2 loadXY = DTid.xy * TileSize;
    uint2 storeXY = DTid.xy;

    float3 maxVelocity = float3(0, 0, 0); // x, y, |v|²

    [unroll]
    for (uint y = 0; y < TileSize; y++)
    {
        [unroll]
        for (uint x = 0; x < TileSize; x++)
        {
            uint2 pixel = loadXY + uint2(x, y);

            if (pixel.x >= gTexSize.x || pixel.y >= gTexSize.y)
                continue;

            float2 v = velocityBuffer.Load(int3(pixel, 0));
            float mag2 = dot(v, v);

            if (mag2 > maxVelocity.z)
            {
                maxVelocity = float3(v, mag2);
            }
        }
    }

    tileMaxOut[storeXY]  = maxVelocity.xy;
}*/

[numthreads(8, 8, 1)]
void CS_TileMax(uint3 DTid : SV_DispatchThreadID)
{
    uint2 tileID = DTid.xy;

    uint2 basePixel = tileID * 8;

    float3 maxVel = float3(0, 0, 0);

    for (uint y = 0; y < 8; y++)
        for (uint x = 0; x < 8; x++)
        {
            uint2 p = basePixel + uint2(x, y);
            if (p.x >= gTexSize.x || p.y >= gTexSize.y)
                continue;

            float2 v = velocityBuffer[p];
            float mag = dot(v, v);

            if (mag > maxVel.z)
                maxVel = float3(v, mag);
        }

    tileMaxOut[tileID] = maxVel.xy;
}

//////////////////////////////////////////////////////

// tile max input (SRV)
Texture2D<float2> tileMaxIn : register(t3);

// neighbour max output (UAV)
RWTexture2D<float2> neighbourMaxOut : register(u1);

// size of tile grid (width/TileSize, height/TileSize)
cbuffer NeighbourCB : register(b2)
{
    int2 gTileCount; // {tilesX, tilesY}
};

//[numthreads(BlockSizeX, BlockSizeY, 1)]
// делать усреднение по тайлу или же для каждого пикселя????????


/*[numthreads(1, 1, 1)]
void CS_NeighbourMax(uint3 id : SV_DispatchThreadID)
{
    int2 tileID = id.xy;

    float3 best = float3(0, 0, 0);

    // search region
    for (int j = -NMRadius; j <= NMRadius; ++j)
    {
        for (int i = -NMRadius; i <= NMRadius; ++i)
        {
            int2 n = tileID + int2(i, j);

            n = clamp(n, int2(0, 0), gTileCount - 1);

            float2 v = tileMaxIn[n];
            float m = dot(v, v);

            if (m > best.z)
                best = float3(v, m);
        }
    }

    neighbourMaxOut[tileID] = best.xy;
}*/

[numthreads(8, 8, 1)]
void CS_NeighbourMax(uint3 DTid : SV_DispatchThreadID)
{
    int2 loadXY = int2(DTid.xy);
    int2 storeXY = loadXY;

    float3 maxVelocity = float3(0, 0, 0);
    float3 initialVelocity;

    // Берём текущую скорость из tileMax
    initialVelocity.xy = tileMaxIn[loadXY];
    initialVelocity.z = dot(initialVelocity.xy, initialVelocity.xy);

    const int STEPS = 5;

    // Перебираем соседние тайлы
    for (int j = -STEPS; j <= STEPS; ++j)
    {
        for (int i = -STEPS; i <= STEPS; ++i)
        {
            int2 n = loadXY + int2(i, j);

            // clamp по экрану / размеру тайлов
            n = clamp(n, int2(0, 0), gTileCount - 1);

            float2 v = tileMaxIn[n];
            float m = dot(v, v);

            if (m > maxVelocity.z)
                maxVelocity = float3(v, m);
        }
    }

    // Корректировка ориентации
    float factor = saturate(initialVelocity.z / maxVelocity.z);

    if (factor > 0.01f)
    {
        maxVelocity.xy = lerp(maxVelocity.xy, normalize(initialVelocity.xy) * sqrt(maxVelocity.z), factor);
    }

    neighbourMaxOut[storeXY] = maxVelocity.xy;
}


/////////

Texture2D<float4> HdrSource : register(t4);
RWTexture2D<float4> HdrTarget : register(u3);
Texture2D<float2> neighbourMaxIn : register(t5);
Texture2D<float2> velocityBuf : register(t6);
Texture2D<float> depth : register(t7);


/// добавить позже
//DSRClampTexCoord
//DSRWithinViewport
//NoiseTexture

cbuffer MotionBlurFinalStageCB : register(b3)
{
    float2 InvResolution; // 1 / width, 1 / height
    float MaxVelocity;
    float Padding;
};

 
[numthreads(BlockSizeX, BlockSizeY, 1)]
void CS_motionBlurFinalTest(
    uint3 groupId : SV_GroupID,
    uint3 groupThreadId : SV_GroupThreadID)
{
   
    
   /* int2 pixelCoord = int2(
        groupId.x * BlockSizeX + groupThreadId.x,
        groupId.y * BlockSizeY + groupThreadId.y
    );

    float2 uv = (float2(pixelCoord) + 0.5f) * InvResolution;

    // Тестовое чтение всех текстур
    float4 hdrColor = HdrSource.Load(int3(pixelCoord, 0));
    float2 neighMax = neighbourMaxIn.Load(int3(pixelCoord, 0));
    float2 vel = velocityBuf.Load(int3(pixelCoord, 0));
    float depthVal = depth.Load(int3(pixelCoord, 0));

    // Тестовая комбинация — просто сложим всё в цвет
    float3 finalColor = hdrColor.rgb + float3(neighMax, 0.0) + float3(vel, 0.0) + depthVal.xxx;

    HdrTarget[pixelCoord] = float4(finalColor, 1.0f);*/
}


// Ограничение скорости
float2 ClampVelocity(float2 velocity, float maxVelocity)
{
    float len = length(velocity);
    return (len > 0.001) ? min(len, maxVelocity) * (velocity / len) : float2(0, 0);
}

// Функции soft blending
float cone(float2 pX, float2 pY, float2 v)
{
    return saturate(1 - distance(pX, pY) / length(v));
}

float cylinder(float2 pX, float2 pY, float2 v)
{
    float L = length(v);
    float D = distance(pX, pY);
    return 1 - smoothstep(0.95f * L, 1.05f * L, D);
}

float softDepthCompare(float za, float zb)
{
    const float SOFT_Z_EXTENT = 0.01; //0.01;
    return saturate(1 - (zb - za) / SOFT_Z_EXTENT);
}

float2 ClampUVTile(float2 uv, int2 pixelCoord)
{
    float2 tileSizeUV = float2(TileSizeDebug / gTexSizeV.x, TileSizeDebug / gTexSizeV.y);
    int2 tileIndex = int2(pixelCoord / TileSizeDebug); // определяем, в каком тайле пиксель

    float2 marginUV = float2(1.0 / gTexSizeV.x, 1.0 / gTexSizeV.y);
    float2 tileMin = float2(tileIndex) * tileSizeUV + marginUV;
    float2 tileMax = tileMin + tileSizeUV - 2.0f * marginUV;

    return clamp(uv, tileMin, tileMax);
}

// Основной расчёт Motion Blur
float3 CalcMotionBlur(float2 uv, int2 pixelCoord)
{    
    float eps = 1.0 / 8192.0f;

    float2 vn = neighbourMaxIn.SampleLevel(samLinear, uv, 0);
    vn = ClampVelocity(vn, MaxVelocity);
    
    float2 vx = velocityBuf.SampleLevel(samPoint, uv, 0);
    float3 cx = HdrSource.SampleLevel(samPoint, uv, 0).rgb;
    float zx = LinearizeDepthV(depth.SampleLevel(samPoint, uv, 0).r);

    //return float3(length(vn), 0, 0);
    //return float3(vn, 1);
    //return float3(vx, 1);
    //return cx;
    //return float3(depth.SampleLevel(samPoint, uv, 0).r, 0, 0);
    
    //float threshold = 0.001f; //1 / 4096.0f;
    float threshold = 1 / 4096.0f;
    
    
    if (length(vn) < threshold)
        return cx;
    //return float3(0, 0, 0);

    float noiseValue = 0;
    
    float weight = 1.0 / (length(vx) + eps);
    float3 sum = cx * weight;

    int STEPS = 10;
    float invSteps = 1.0 / STEPS;

    [loop]
    for (int i = -STEPS; i <= STEPS; i++)
    {
        if (i == 0)
            continue;

        float t = ((float) i + noiseValue) / (float)(STEPS) / 2.0f;
        t *= 3;
        
        float2 sampleUV = uv + (vn / gTexSizeV) * t;
        sampleUV = clamp(sampleUV, 0.0, 1.0);
        //sampleUV = ClampUVTile(sampleUV, pixelCoord); // это точно нужно?

        float3 cy = HdrSource.SampleLevel(samPoint, sampleUV, 0).rgb;
        float zy = LinearizeDepthV(depth.SampleLevel(samPoint, sampleUV, 0).r);
        float2 vy = velocityBuf.SampleLevel(samPoint, sampleUV, 0);

        float f = softDepthCompare(zx, zy);
        float b = softDepthCompare(zy, zx);

        float ay = f * cone(sampleUV, uv, vy)
                 + b * cone(uv, sampleUV, vx)
                 + cylinder(sampleUV, uv, vy) * cylinder(uv, sampleUV, vx) * 2.0;

        weight += ay;
        sum += cy * ay;
    }

    return sum / weight;
}

// -----------------------------------
// Compute Shader Entry
// -----------------------------------
[numthreads(BlockSizeX, BlockSizeY, 1)]
void CS_motionBlurFinal(uint3 DTid : SV_DispatchThreadID)
{
    int2 pixelCoord = int2(DTid.xy);
    
    if ((pixelCoord.x >= gTexSizeV.x)  || (pixelCoord.y >= gTexSizeV.y) || (pixelCoord.x < 0) || (pixelCoord.y < 0))
        return;
    
    float2 uv = (float2(pixelCoord) + 0.5f) * (1.0f / float2(gTexSizeV));
    
    float3 finalColor = CalcMotionBlur(uv, pixelCoord);
    HdrTarget[pixelCoord] = float4(finalColor, 1.0f);
}

/*
[numthreads(BlockSizeX, BlockSizeY, 1)]
void CS_motionBlurFinalTest2(uint3 groupId : SV_GroupID, uint3 groupThreadId : SV_GroupThreadID)
{
    int2 pixelCoord = int2(
        groupId.x * BlockSizeX + groupThreadId.x,
        groupId.y * BlockSizeY + groupThreadId.y
    );

    float3 finalColor = CalcMotionBlur2(pixelCoord);
    HdrTarget[pixelCoord] = float4(finalColor, 1.0f);
}*/