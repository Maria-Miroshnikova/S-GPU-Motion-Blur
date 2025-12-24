//***************************************************************************************
// SsaoApp.cpp by Frank Luna (C) 2015 All Rights Reserved.
//***************************************************************************************

#include "../../Common/d3dApp.h"
#include "../../Common/MathHelper.h"
#include "../../Common/UploadBuffer.h"
#include "../../Common/GeometryGenerator.h"
#include "../../Common/Camera.h"
#include "FrameResource.h"
#include "ShadowMap.h"
#include "Ssao.h"

// MB additions
#include "dxcapi.h"
#include "wrl.h"

using Microsoft::WRL::ComPtr;
using namespace DirectX;
using namespace DirectX::PackedVector;

const int gNumFrameResources = 3;

// Lightweight structure stores parameters to draw a shape.  This will
// vary from app-to-app.
struct RenderItem
{
	RenderItem() = default;
    RenderItem(const RenderItem& rhs) = delete;
 
    // World matrix of the shape that describes the object's local space
    // relative to the world space, which defines the position, orientation,
    // and scale of the object in the world.
    XMFLOAT4X4 World = MathHelper::Identity4x4();

	XMFLOAT4X4 TexTransform = MathHelper::Identity4x4();

	// Dirty flag indicating the object data has changed and we need to update the constant buffer.
	// Because we have an object cbuffer for each FrameResource, we have to apply the
	// update to each FrameResource.  Thus, when we modify obect data we should set 
	// NumFramesDirty = gNumFrameResources so that each frame resource gets the update.
	int NumFramesDirty = gNumFrameResources;

	// Index into GPU constant buffer corresponding to the ObjectCB for this render item.
	UINT ObjCBIndex = -1;

	Material* Mat = nullptr;
	MeshGeometry* Geo = nullptr;

    // Primitive topology.
    D3D12_PRIMITIVE_TOPOLOGY PrimitiveType = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST;

    // DrawIndexedInstanced parameters.
    UINT IndexCount = 0;
    UINT StartIndexLocation = 0;
    int BaseVertexLocation = 0;
};

enum class RenderLayer : int
{
	Opaque = 0,
    Debug,
	Sky,
	Count
};

class SsaoApp : public D3DApp
{
public:
    SsaoApp(HINSTANCE hInstance);
    SsaoApp(const SsaoApp& rhs) = delete;
    SsaoApp& operator=(const SsaoApp& rhs) = delete;
    ~SsaoApp();

    virtual bool Initialize()override;

private:
    virtual void CreateRtvAndDsvDescriptorHeaps()override;
    virtual void OnResize()override;
    virtual void Update(const GameTimer& gt)override;
    virtual void Draw(const GameTimer& gt)override;

    virtual void OnMouseDown(WPARAM btnState, int x, int y)override;
    virtual void OnMouseUp(WPARAM btnState, int x, int y)override;
    virtual void OnMouseMove(WPARAM btnState, int x, int y)override;

    void OnKeyboardInput(const GameTimer& gt);
	void AnimateMaterials(const GameTimer& gt);
	void UpdateObjectCBs(const GameTimer& gt);
	void UpdateMaterialBuffer(const GameTimer& gt);
    void UpdateShadowTransform(const GameTimer& gt);
	void UpdateMainPassCB(const GameTimer& gt);
    void UpdateShadowPassCB(const GameTimer& gt);
    void UpdateSsaoCB(const GameTimer& gt);

	void LoadTextures();
    void BuildRootSignature();
    // TODO: motion blur
    void BuildSsaoRootSignature();
	void BuildDescriptorHeaps();
    void BuildShadersAndInputLayout();
    void BuildShapeGeometry();
    void BuildSkullGeometry();
    void BuildPSOs();
    void BuildFrameResources();
    void BuildMaterials();
    void BuildRenderItems();
    void DrawRenderItems(ID3D12GraphicsCommandList* cmdList, const std::vector<RenderItem*>& ritems);
    void DrawSceneToShadowMap();
	void DrawNormalsAndDepth();

    CD3DX12_CPU_DESCRIPTOR_HANDLE GetCpuSrv(int index)const;
    CD3DX12_GPU_DESCRIPTOR_HANDLE GetGpuSrv(int index)const;
    CD3DX12_CPU_DESCRIPTOR_HANDLE GetDsv(int index)const;
    CD3DX12_CPU_DESCRIPTOR_HANDLE GetRtv(int index)const;

	std::array<const CD3DX12_STATIC_SAMPLER_DESC, 7> GetStaticSamplers();

    // motion blur часть

    XMFLOAT4X4 matrixView_prev = MathHelper::Identity4x4();
    //XMFLOAT4X4 mPrevProj = MathHelper::Identity4x4();
    //XMFLOAT4X4 mPrevViewProj = MathHelper::Identity4x4();

    void update_MB_buffers(const GameTimer& gt);
    Microsoft::WRL::ComPtr<ID3D12Resource> mDepthStencilBuffer_prev;
    void onResize_mb();

    UINT mPrevDepthSrvIndex = 0;
    UINT mCurDepthSrvIndex = 0;
    void copyDepthCurrToPrev();

    Microsoft::WRL::ComPtr<ID3D12Resource> mVelocityBuffer = nullptr;
    UINT mVelocitySrvIndex = 0;
    UINT mVelocityRtvIndex = 0;
    UINT mVelocityUavIndex = 0;

    void BuildRootSignature_mb();
    ComPtr<ID3D12RootSignature> mRootSignature_velocity_mb = nullptr;
    ComPtr<ID3D12RootSignature> mRootSignature_velocity_cs_mb = nullptr;

    void BuildPSO_mb(D3D12_GRAPHICS_PIPELINE_STATE_DESC basePsoDesc);
    void DrawVelocity(ID3D12GraphicsCommandList* cmdList);

    UINT tileSize = 8;
    Microsoft::WRL::ComPtr<ID3D12Resource> mTileMaxBuffer = nullptr;
    UINT mTileMaxSrvIndex = 0;
    UINT mTileMaxUavIndex = 0;
    UINT mTileMaxRtvIndex = 0;

    ComPtr<ID3D12RootSignature> mRootSignature_tileMax_mb = nullptr;

    /*
    struct TileMaxCB
    {
        XMFLOAT2 gTexSize;
        XMFLOAT2 gInvTexSize;
        int TileSize;
        float padding;   // must remain! 16-byte alignment
    };*/

    // это в FrameResources объявлено
    //Microsoft::WRL::ComPtr<ID3D12Resource> tileMaxCB;

    void DrawTileMax(ID3D12GraphicsCommandList* cmdList);

    Microsoft::WRL::ComPtr<ID3D12Resource> mNeighbourMaxBuffer = nullptr;
    UINT mNeighbourMaxSrvIndex = 0;
    UINT mNeighbourMaxUavIndex = 0;

    ComPtr<ID3D12RootSignature> mRootSignature_neighbourMax_mb = nullptr;

    void DrawNeighbourMax(ID3D12GraphicsCommandList* cmdList);

    // компилятор для cs_6_0 для потоков
    ComPtr<IDxcBlob> CompileShaderSM6(
        const std::wstring& filename,
        const std::wstring& entryPoint,
        const std::wstring& target);

    // копирнуть кадр сюда перед вызовом мошн блюра
    Microsoft::WRL::ComPtr<ID3D12Resource> mHdrBeforeMbBuffer = nullptr;
    UINT mHdrBeforeMbSrvIndex = 0;
    UINT mHdrBeforeMbRtvIndex = 0;

    Microsoft::WRL::ComPtr<ID3D12Resource> mHdrAfterMbBuffer = nullptr;
    UINT mHdrAfterMbSrvIndex = 0;
    UINT mHdrAfterMbUavIndex = 0;

    ComPtr<ID3D12RootSignature> mRootSignature_motionBlurFinal_mb = nullptr;
    
    void DrawMotionBlurFinal(ID3D12GraphicsCommandList* cmdList);

private:

    std::vector<std::unique_ptr<FrameResource>> mFrameResources;
    FrameResource* mCurrFrameResource = nullptr;
    int mCurrFrameResourceIndex = 0;

    ComPtr<ID3D12RootSignature> mRootSignature = nullptr;
    ComPtr<ID3D12RootSignature> mSsaoRootSignature = nullptr;

	ComPtr<ID3D12DescriptorHeap> mSrvDescriptorHeap = nullptr;

	std::unordered_map<std::string, std::unique_ptr<MeshGeometry>> mGeometries;
	std::unordered_map<std::string, std::unique_ptr<Material>> mMaterials;
	std::unordered_map<std::string, std::unique_ptr<Texture>> mTextures;
	std::unordered_map<std::string, ComPtr<ID3DBlob>> mShaders;
	std::unordered_map<std::string, ComPtr<ID3D12PipelineState>> mPSOs;

    std::vector<D3D12_INPUT_ELEMENT_DESC> mInputLayout;
 
	// List of all the render items.
	std::vector<std::unique_ptr<RenderItem>> mAllRitems;

	// Render items divided by PSO.
	std::vector<RenderItem*> mRitemLayer[(int)RenderLayer::Count];

	UINT mSkyTexHeapIndex = 0;
    UINT mShadowMapHeapIndex = 0;
    UINT mSsaoHeapIndexStart = 0;
    UINT mSsaoAmbientMapIndex = 0;

    UINT mNullCubeSrvIndex = 0;
    UINT mNullTexSrvIndex1 = 0;
    UINT mNullTexSrvIndex2 = 0;

    CD3DX12_GPU_DESCRIPTOR_HANDLE mNullSrv;

    PassConstants mMainPassCB;  // index 0 of pass cbuffer.
    PassConstants mShadowPassCB;// index 1 of pass cbuffer.

	Camera mCamera;

    std::unique_ptr<ShadowMap> mShadowMap;

    std::unique_ptr<Ssao> mSsao;

    DirectX::BoundingSphere mSceneBounds;

    float mLightNearZ = 0.0f;
    float mLightFarZ = 0.0f;
    XMFLOAT3 mLightPosW;
    XMFLOAT4X4 mLightView = MathHelper::Identity4x4();
    XMFLOAT4X4 mLightProj = MathHelper::Identity4x4();
    XMFLOAT4X4 mShadowTransform = MathHelper::Identity4x4();

    float mLightRotationAngle = 0.0f;
    XMFLOAT3 mBaseLightDirections[3] = {
        XMFLOAT3(0.57735f, -0.57735f, 0.57735f),
        XMFLOAT3(-0.57735f, -0.57735f, 0.57735f),
        XMFLOAT3(0.0f, -0.707f, -0.707f)
    };
    XMFLOAT3 mRotatedLightDirections[3];

    POINT mLastMousePos;
};

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE prevInstance,
    PSTR cmdLine, int showCmd)
{
    // Enable run-time memory check for debug builds.
#if defined(DEBUG) | defined(_DEBUG)
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

    try
    {
        SsaoApp theApp(hInstance);
        if(!theApp.Initialize())
            return 0;

        return theApp.Run();
    }
    catch(DxException& e)
    {
        MessageBox(nullptr, e.ToString().c_str(), L"HR Failed", MB_OK);
        return 0;
    }
}

SsaoApp::SsaoApp(HINSTANCE hInstance)
    : D3DApp(hInstance)
{
    // Estimate the scene bounding sphere manually since we know how the scene was constructed.
    // The grid is the "widest object" with a width of 20 and depth of 30.0f, and centered at
    // the world space origin.  In general, you need to loop over every world space vertex
    // position and compute the bounding sphere.
    mSceneBounds.Center = XMFLOAT3(0.0f, 0.0f, 0.0f);
    mSceneBounds.Radius = sqrtf(10.0f*10.0f + 15.0f*15.0f);
}

SsaoApp::~SsaoApp()
{
    if(md3dDevice != nullptr)
        FlushCommandQueue();
}

bool SsaoApp::Initialize()
{
    if(!D3DApp::Initialize())
        return false;

    // Reset the command list to prep for initialization commands.
    ThrowIfFailed(mCommandList->Reset(mDirectCmdListAlloc.Get(), nullptr));

	mCamera.SetPosition(0.0f, 2.0f, -15.0f);
 
    mShadowMap = std::make_unique<ShadowMap>(md3dDevice.Get(),
        2048, 2048);

    // TODO: add Motion Blur

    mSsao = std::make_unique<Ssao>(
        md3dDevice.Get(),
        mCommandList.Get(),
        mClientWidth, mClientHeight);

	LoadTextures();
    BuildRootSignature();
    
    // MB чать
    BuildRootSignature_mb();

    BuildSsaoRootSignature();
	BuildDescriptorHeaps();
    BuildShadersAndInputLayout();
    BuildShapeGeometry();
    BuildSkullGeometry();
	BuildMaterials();
    BuildRenderItems();
    BuildFrameResources();
    BuildPSOs();

    mSsao->SetPSOs(mPSOs["ssao"].Get(), mPSOs["ssaoBlur"].Get());

    // Execute the initialization commands.
    ThrowIfFailed(mCommandList->Close());
    ID3D12CommandList* cmdsLists[] = { mCommandList.Get() };
    mCommandQueue->ExecuteCommandLists(_countof(cmdsLists), cmdsLists);

    // Wait until initialization is complete.
    FlushCommandQueue();

    return true;
}

void SsaoApp::CreateRtvAndDsvDescriptorHeaps()
{
    // Add +1 for screen normal map, +2 for ambient maps
    // + 1 для velocity mb
    // + 1 для hdr buffer
    D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc;
    rtvHeapDesc.NumDescriptors = SwapChainBufferCount + 5;
    rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    rtvHeapDesc.NodeMask = 0;
    ThrowIfFailed(md3dDevice->CreateDescriptorHeap(
        &rtvHeapDesc, IID_PPV_ARGS(mRtvHeap.GetAddressOf())));

    // Add +1 DSV for shadow map.
    D3D12_DESCRIPTOR_HEAP_DESC dsvHeapDesc;
    dsvHeapDesc.NumDescriptors = 2;
    dsvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    dsvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    dsvHeapDesc.NodeMask = 0;
    ThrowIfFailed(md3dDevice->CreateDescriptorHeap(
        &dsvHeapDesc, IID_PPV_ARGS(mDsvHeap.GetAddressOf())));
}
 
void SsaoApp::OnResize()
{
    D3DApp::OnResize();

	mCamera.SetLens(0.25f*MathHelper::Pi, AspectRatio(), 1.0f, 1000.0f);

    // TODO: motion blur

    if(mSsao != nullptr)
    {
        mSsao->OnResize(mClientWidth, mClientHeight);

        // Resources changed, so need to rebuild descriptors.
        mSsao->RebuildDescriptors(mDepthStencilBuffer.Get());
    }

    onResize_mb();
}

void SsaoApp::onResize_mb() {
    
    // второй буффер для хранения прошлой глубины
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Alignment = 0;
    desc.Width = mClientWidth;
    desc.Height = mClientHeight;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_R24G8_TYPELESS; // как в основном depth
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = D3D12_RESOURCE_FLAG_NONE; // ← ВАЖНО!

    ThrowIfFailed(md3dDevice->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_HEAP_FLAG_NONE,
        &desc,
        D3D12_RESOURCE_STATE_COMMON,
        nullptr,
        IID_PPV_ARGS(&mDepthStencilBuffer_prev)
    ));

    // SRV создаем в builddesctiprorsheap

    // создание буффера скоростей

    D3D12_RESOURCE_DESC texDesc = {};
    texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texDesc.Alignment = 0;
    texDesc.Width = mClientWidth;     // как экран
    texDesc.Height = mClientHeight;
    texDesc.DepthOrArraySize = 1;
    texDesc.MipLevels = 1;
    texDesc.Format = DXGI_FORMAT_R16G16_FLOAT; // высокоточный velocity buffer
    texDesc.SampleDesc.Count = 1;   // БЕЗ MSAA!
    texDesc.SampleDesc.Quality = 0;
    texDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    texDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;

    D3D12_CLEAR_VALUE clear;
    clear.Format = DXGI_FORMAT_R16G16_FLOAT;
    clear.Color[0] = 0.0f;
    clear.Color[1] = 0.0f;
    clear.Color[2] = 0.0f;
    clear.Color[3] = 0.0f;

    ThrowIfFailed(md3dDevice->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_HEAP_FLAG_NONE,
        &texDesc,
        D3D12_RESOURCE_STATE_COMMON,
        &clear,
        IID_PPV_ARGS(&mVelocityBuffer)));

    // создание tileMax буфера

    UINT tileW = (mClientWidth + tileSize - 1) / tileSize;
    UINT tileH = (mClientHeight + tileSize - 1) / tileSize;

    D3D12_RESOURCE_DESC descTileMax = {};
    descTileMax.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    descTileMax.Width = tileW;
    descTileMax.Height = tileH;
    descTileMax.DepthOrArraySize = 1;
    descTileMax.MipLevels = 1;
    descTileMax.Format = DXGI_FORMAT_R16G16_FLOAT;
    descTileMax.SampleDesc.Count = 1;
    descTileMax.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    FLOAT clearValue[4] = { 0,0,0,0 };

    D3D12_CLEAR_VALUE clearTileMax = {};
    clearTileMax.Format = DXGI_FORMAT_R16G16_FLOAT;
    memcpy(clearTileMax.Color, clearValue, sizeof(clearValue));

    ThrowIfFailed(md3dDevice->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_HEAP_FLAG_NONE,
        &descTileMax,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        nullptr,       // <--- ВАЖНО!!! ДОЛЖНО БЫТЬ nullptr
        IID_PPV_ARGS(&mTileMaxBuffer)
    ));

    // создание констант с размерами для tilemax делается в FrameResources!

    // neighbour max буффер

    D3D12_RESOURCE_DESC descNeighbourMax = {};
    descNeighbourMax.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    descNeighbourMax.Width = tileW;
    descNeighbourMax.Height = tileH;
    descNeighbourMax.DepthOrArraySize = 1;
    descNeighbourMax.MipLevels = 1;
    descNeighbourMax.Format = DXGI_FORMAT_R16G16_FLOAT;
    descNeighbourMax.SampleDesc.Count = 1;
    descNeighbourMax.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

//    FLOAT clearValue[4] = { 0,0,0,0 };

    D3D12_CLEAR_VALUE clearNeighbourMax = {};
    clearNeighbourMax.Format = DXGI_FORMAT_R16G16_FLOAT;
    memcpy(clearNeighbourMax.Color, clearValue, sizeof(clearValue));

    ThrowIfFailed(md3dDevice->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_HEAP_FLAG_NONE,
        &descNeighbourMax,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        nullptr,       // <--- ВАЖНО!!! ДОЛЖНО БЫТЬ nullptr
        IID_PPV_ARGS(&mNeighbourMaxBuffer)
    ));

    // mHdrBeforeMbBuffer 
    D3D12_RESOURCE_DESC descHdr = {};
    descHdr.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    descHdr.Alignment = 0;
    descHdr.Width = mClientWidth;
    descHdr.Height = mClientHeight;
    descHdr.DepthOrArraySize = 1;
    descHdr.MipLevels = 1;
    descHdr.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    descHdr.SampleDesc.Count = 1;
    descHdr.SampleDesc.Quality = 0;
    descHdr.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    descHdr.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;

    ThrowIfFailed(md3dDevice->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_HEAP_FLAG_NONE,
        &descHdr,
        D3D12_RESOURCE_STATE_RENDER_TARGET,
        nullptr,
        IID_PPV_ARGS(&mHdrBeforeMbBuffer)
    ));

    //mHdrAfterMbBuffer
    D3D12_RESOURCE_DESC descHdrAfter = {};
    descHdrAfter.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    descHdrAfter.Alignment = 0;
    descHdrAfter.Width = mClientWidth;
    descHdrAfter.Height = mClientHeight;
    descHdrAfter.DepthOrArraySize = 1;
    descHdrAfter.MipLevels = 1;
    descHdrAfter.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    descHdrAfter.SampleDesc.Count = 1;
    descHdrAfter.SampleDesc.Quality = 0;
    descHdrAfter.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    descHdrAfter.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    ThrowIfFailed(md3dDevice->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_HEAP_FLAG_NONE,
        &descHdrAfter,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        nullptr,
        IID_PPV_ARGS(&mHdrAfterMbBuffer)
    ));
}

void SsaoApp::Update(const GameTimer& gt)
{
    OnKeyboardInput(gt);

    // Cycle through the circular frame resource array.
    mCurrFrameResourceIndex = (mCurrFrameResourceIndex + 1) % gNumFrameResources;
    mCurrFrameResource = mFrameResources[mCurrFrameResourceIndex].get();

    // Has the GPU finished processing the commands of the current frame resource?
    // If not, wait until the GPU has completed commands up to this fence point.
    if(mCurrFrameResource->Fence != 0 && mFence->GetCompletedValue() < mCurrFrameResource->Fence)
    {
        HANDLE eventHandle = CreateEventEx(nullptr, false, false, EVENT_ALL_ACCESS);
        ThrowIfFailed(mFence->SetEventOnCompletion(mCurrFrameResource->Fence, eventHandle));
        WaitForSingleObject(eventHandle, INFINITE);
        CloseHandle(eventHandle);
    }

    //
    // Animate the lights (and hence shadows).
    //

    mLightRotationAngle += 0.1f*gt.DeltaTime();

    XMMATRIX R = XMMatrixRotationY(mLightRotationAngle);
    for(int i = 0; i < 3; ++i)
    {
        XMVECTOR lightDir = XMLoadFloat3(&mBaseLightDirections[i]);
        lightDir = XMVector3TransformNormal(lightDir, R);
        XMStoreFloat3(&mRotatedLightDirections[i], lightDir);
    }

	AnimateMaterials(gt);
	UpdateObjectCBs(gt);
	UpdateMaterialBuffer(gt);
    UpdateShadowTransform(gt);
	UpdateMainPassCB(gt);
    UpdateShadowPassCB(gt);
    UpdateSsaoCB(gt);
    
    // МB часть
    update_MB_buffers(gt);
}

// обновляем буффер с матрицами для подсчета буффера скоростей
void SsaoApp::update_MB_buffers(const GameTimer& gt) {
    
    // velocity
    auto currFR = mCurrFrameResource;

    MB_velocityStage_CB velocityStage_cb;

    XMMATRIX view = mCamera.GetView();
    XMMATRIX proj = mCamera.GetProj();
    XMMATRIX prevView = XMLoadFloat4x4(&matrixView_prev);

    XMMATRIX currViewProj = XMMatrixMultiply(view, proj);
    XMMATRIX prevViewProj = XMMatrixMultiply(prevView, proj);
    XMMATRIX invCurrViewProj = XMMatrixInverse(nullptr, currViewProj);
    XMMATRIX reprojection =
        XMMatrixMultiply(prevViewProj, invCurrViewProj);


    XMMATRIX invProj = XMMatrixInverse(nullptr, proj);
    XMMATRIX viewInv = XMMatrixInverse(nullptr, view);
    XMMATRIX viewInvPrev = XMMatrixInverse(nullptr, prevView);

    velocityStage_cb.gTexSizeV = XMINT2(mClientWidth, mClientHeight);
    velocityStage_cb.gNearFar = XMFLOAT4(mCamera.GetNearZ(), mCamera.GetFarZ(), 0, 0); //XMFLOAT4(0, 1000, 0, 0); //

    XMStoreFloat4x4(&velocityStage_cb.InverseProjMatrix_, XMMatrixTranspose(invProj));
    XMStoreFloat4x4(&velocityStage_cb.viewMatrixInvCur_, XMMatrixTranspose(viewInv));
    XMStoreFloat4x4(&velocityStage_cb.viewMatrixInvPrev_, XMMatrixTranspose(viewInvPrev));
    XMStoreFloat4x4(&velocityStage_cb.ReprojectionMatrix_, XMMatrixTranspose(reprojection));

    currFR->MotionBlurCB->CopyData(0, velocityStage_cb);

    // tilemax

    MB_tileMaxStage_CB cb;
    cb.gTexSize = XMFLOAT2((float)mClientWidth, (float)mClientHeight);
    cb.gInvTexSize = XMFLOAT2(1.0f / mClientWidth, 1.0f / mClientHeight);
    cb.TileSize = tileSize;
    cb.padding = 0;

    currFR->tileMaxCB->CopyData(0, cb);

    // neighboutMax

    MB_neighbourMaxStage_CB neigbourMaxStage_cb;
    neigbourMaxStage_cb.gTileCount = XMFLOAT2 (mClientWidth * 1.0f / tileSize, mClientHeight * 1.0f / tileSize);

    currFR->NeighbourMaxCB->CopyData(0, neigbourMaxStage_cb);

    // mbFinalStage

    MB_motionBlurFinalStage_CB final_cb;
    final_cb.InvResolution = XMFLOAT2 (1.0f / tileSize, 1.0f / tileSize);
    final_cb.MaxVelocity = 30.0f; // или подбирать визуально
    currFR->MotionBlurFinalCB->CopyData(0, final_cb);
}

void SsaoApp::Draw(const GameTimer& gt)
{
    auto cmdListAlloc = mCurrFrameResource->CmdListAlloc;

    // Reuse the memory associated with command recording.
    // We can only reset when the associated command lists have finished execution on the GPU.
    ThrowIfFailed(cmdListAlloc->Reset());

    // A command list can be reset after it has been added to the command queue via ExecuteCommandList.
    // Reusing the command list reuses memory.
    ThrowIfFailed(mCommandList->Reset(cmdListAlloc.Get(), mPSOs["opaque"].Get()));

    ID3D12DescriptorHeap* descriptorHeaps[] = { mSrvDescriptorHeap.Get() };
    mCommandList->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);

    mCommandList->SetGraphicsRootSignature(mRootSignature.Get());

	//
	// Shadow map pass.
	//

    // Bind all the materials used in this scene.  For structured buffers, we can bypass the heap and 
    // set as a root descriptor.
    auto matBuffer = mCurrFrameResource->MaterialBuffer->Resource();
    mCommandList->SetGraphicsRootShaderResourceView(2, matBuffer->GetGPUVirtualAddress());
	
    // Bind null SRV for shadow map pass.
    mCommandList->SetGraphicsRootDescriptorTable(3, mNullSrv);	 

    // Bind all the textures used in this scene.  Observe
    // that we only have to specify the first descriptor in the table.  
    // The root signature knows how many descriptors are expected in the table.
    mCommandList->SetGraphicsRootDescriptorTable(4, mSrvDescriptorHeap->GetGPUDescriptorHandleForHeapStart());

    DrawSceneToShadowMap();

	//
	// Normal/depth pass.
	//
	
	DrawNormalsAndDepth();
	
	
	// Compute SSAO.
	// 
	
    mCommandList->SetGraphicsRootSignature(mSsaoRootSignature.Get());
    mSsao->ComputeSsao(mCommandList.Get(), mCurrFrameResource, 3);
	
    // MB часть

    mCommandList->SetGraphicsRootSignature(mRootSignature_velocity_mb.Get());
    DrawVelocity(mCommandList.Get());

    mCommandList->SetGraphicsRootSignature(mRootSignature_tileMax_mb.Get());
    DrawTileMax(mCommandList.Get());   

    // так по тайлам делать усреднение или по каждому пиксели
    mCommandList->SetGraphicsRootSignature(mRootSignature_neighbourMax_mb.Get());
    DrawNeighbourMax(mCommandList.Get());
    
    //
	// Main rendering pass.
	//
	
    mCommandList->SetGraphicsRootSignature(mRootSignature.Get());

    // Rebind state whenever graphics root signature changes.

    // Bind all the materials used in this scene.  For structured buffers, we can bypass the heap and 
    // set as a root descriptor.
    matBuffer = mCurrFrameResource->MaterialBuffer->Resource();
    mCommandList->SetGraphicsRootShaderResourceView(2, matBuffer->GetGPUVirtualAddress());


    mCommandList->RSSetViewports(1, &mScreenViewport);
    mCommandList->RSSetScissorRects(1, &mScissorRect);

    // Indicate a state transition on the resource usage.
	mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(CurrentBackBuffer(),
		D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET));

    // Clear the back buffer.
    mCommandList->ClearRenderTargetView(CurrentBackBufferView(), Colors::LightSteelBlue, 0, nullptr);

    // WE ALREADY WROTE THE DEPTH INFO TO THE DEPTH BUFFER IN DrawNormalsAndDepth,
    // SO DO NOT CLEAR DEPTH.

    // Specify the buffers we are going to render to.
    // MB: вместо &CurrentBackBufferView() пишем в HdrBeforeMB, потом отрисовываем sky, opaque и т д туда, затем опять
    // выставим &CurrentBackBufferView() и туда отрисуем итоговый motinBlurFinal
    // MB FINAL
    mCommandList->OMSetRenderTargets(1, &GetRtv(mHdrBeforeMbRtvIndex), true, &DepthStencilView());

	// Bind all the textures used in this scene.  Observe
    // that we only have to specify the first descriptor in the table.  
    // The root signature knows how many descriptors are expected in the table.
    mCommandList->SetGraphicsRootDescriptorTable(4, mSrvDescriptorHeap->GetGPUDescriptorHandleForHeapStart());
	
    auto passCB = mCurrFrameResource->PassCB->Resource();
	mCommandList->SetGraphicsRootConstantBufferView(1, passCB->GetGPUVirtualAddress());

    // Bind the sky cube map.  For our demos, we just use one "world" cube map representing the environment
    // from far away, so all objects will use the same cube map and we only need to set it once per-frame.  
    // If we wanted to use "local" cube maps, we would have to change them per-object, or dynamically
    // index into an array of cube maps.

    CD3DX12_GPU_DESCRIPTOR_HANDLE skyTexDescriptor(mSrvDescriptorHeap->GetGPUDescriptorHandleForHeapStart());
    skyTexDescriptor.Offset(mSkyTexHeapIndex, mCbvSrvUavDescriptorSize);
    mCommandList->SetGraphicsRootDescriptorTable(3, skyTexDescriptor);

    mCommandList->SetPipelineState(mPSOs["opaque"].Get());
    DrawRenderItems(mCommandList.Get(), mRitemLayer[(int)RenderLayer::Opaque]);

    //mCommandList->SetPipelineState(mPSOs["debug"].Get());
    //DrawRenderItems(mCommandList.Get(), mRitemLayer[(int)RenderLayer::Debug]);

	mCommandList->SetPipelineState(mPSOs["sky"].Get());
	DrawRenderItems(mCommandList.Get(), mRitemLayer[(int)RenderLayer::Sky]);

    // MB FINAL
    /*// переключаем буфер на копирование
    mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(CurrentBackBuffer(),
        D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_DEST));

    mCommandList->CopyResource(
        CurrentBackBuffer(), // dst = back buffer
        mHdrBeforeMbBuffer.Get()                         // src = hdrBeforeMb
    );
    
    // переключаем буфер опять на рендер таргет
    mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(CurrentBackBuffer(),
        D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_RENDER_TARGET));
    */

    // вызываем mb final

    mCommandList->SetGraphicsRootSignature(mRootSignature_motionBlurFinal_mb.Get());
    DrawMotionBlurFinal(mCommandList.Get());

    mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(CurrentBackBuffer(),
        D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_DEST));

    mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(mHdrAfterMbBuffer.Get(),
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE));

    mCommandList->CopyResource(
        CurrentBackBuffer(), // dst = back buffer
        mHdrAfterMbBuffer.Get()                         // src = hdrBeforeMb
    );

    mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(CurrentBackBuffer(),
        D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_RENDER_TARGET));


    // MB: копируем текущую глубину в прошлую до закрытия списка команд
    copyDepthCurrToPrev();

    // MB вот тут скорее всего будем вызывать моушн блюр и писать в бэкбуффер

    // Indicate a state transition on the resource usage.
	mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(CurrentBackBuffer(),
		D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT));

    // Done recording commands.
    ThrowIfFailed(mCommandList->Close());

    // Add the command list to the queue for execution.
    ID3D12CommandList* cmdsLists[] = { mCommandList.Get() };
    mCommandQueue->ExecuteCommandLists(_countof(cmdsLists), cmdsLists);

    // Swap the back and front buffers
    ThrowIfFailed(mSwapChain->Present(0, 0));
	mCurrBackBuffer = (mCurrBackBuffer + 1) % SwapChainBufferCount;

    // MB: запомнить старую ViewMatrix
    XMStoreFloat4x4(&matrixView_prev, mCamera.GetView());

    // Advance the fence value to mark commands up to this fence point.
    mCurrFrameResource->Fence = ++mCurrentFence;

    // Add an instruction to the command queue to set a new fence point. 
    // Because we are on the GPU timeline, the new fence point won't be 
    // set until the GPU finishes processing all the commands prior to this Signal().
    mCommandQueue->Signal(mFence.Get(), mCurrentFence);
}

void SsaoApp::DrawVelocity(ID3D12GraphicsCommandList* cmdList) {

    assert(mDepthStencilBuffer != nullptr);
    assert(mDepthStencilBuffer_prev != nullptr);
    assert(mVelocityBuffer != nullptr);

    auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(
        mDepthStencilBuffer.Get(),
        D3D12_RESOURCE_STATE_DEPTH_WRITE,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
    );
    cmdList->ResourceBarrier(1, &barrier);

    // версия для графики
    /*

    // 1. Устанавливаем viewport и scissor (тот же, что для экрана)
    cmdList->RSSetViewports(1, &mScreenViewport);
    cmdList->RSSetScissorRects(1, &mScissorRect);

    // 2. Устанавливаем PSO для velocity pass
    cmdList->SetPipelineState(mPSOs["velocity"].Get());

    // 3. Привязываем Root Signature
    cmdList->SetGraphicsRootSignature(mRootSignature_velocity_mb.Get());

    // 4. Передаём CBV с матрицами (current и previous view/proj)
    auto motionBlurCB = mCurrFrameResource->MotionBlurCB->Resource();
    cmdList->SetGraphicsRootConstantBufferView(0, motionBlurCB->GetGPUVirtualAddress());

    // 5. Привязываем SRV на depth буферы
    cmdList->SetGraphicsRootDescriptorTable(1, GetGpuSrv(mCurDepthSrvIndex));   // текущая глубина
    cmdList->SetGraphicsRootDescriptorTable(2, GetGpuSrv(mPrevDepthSrvIndex));  // прошлый depth
   
    // 6. Указываем render target (velocity buffer)
    cmdList->OMSetRenderTargets(1, &GetRtv(mVelocityRtvIndex), false, nullptr);

    // 7. Нарисовать fullscreen треугольник
    cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    cmdList->DrawInstanced(3, 1, 0, 0);

    cmdList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
        mDepthStencilBuffer.Get(),
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_DEPTH_WRITE
    ));

    */

    auto barrierVelocity = CD3DX12_RESOURCE_BARRIER::Transition(
        mVelocityBuffer.Get(),
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    );
    cmdList->ResourceBarrier(1, &barrierVelocity);

    //версия для компьют
    // 
    // 2. Устанавливаем PSO для velocity pass
    cmdList->SetPipelineState(mPSOs["velocityCS"].Get());

    // 3. Привязываем Root Signature
    cmdList->SetComputeRootSignature(mRootSignature_velocity_cs_mb.Get());

    // 4. Передаём CBV с матрицами(current и previous view / proj)
    auto motionBlurCB = mCurrFrameResource->MotionBlurCB->Resource();
    cmdList->SetComputeRootConstantBufferView(0, motionBlurCB->GetGPUVirtualAddress());

    // 5. Привязываем SRV на depth буферы
    cmdList->SetComputeRootDescriptorTable(1, GetGpuSrv(mCurDepthSrvIndex));   // текущая глубина
    cmdList->SetComputeRootDescriptorTable(2, GetGpuSrv(mPrevDepthSrvIndex));  // прошлый depth

    // 6. UAV: velocity target
    cmdList->SetComputeRootDescriptorTable(3, GetGpuSrv(mVelocityUavIndex));

    // Dispatch
    auto BlockSizeX = tileSize; // TODO
    auto BlockSizeY = tileSize; // TODO
    UINT groupX = (mClientWidth + BlockSizeX - 1) / BlockSizeX;
    UINT groupY = (mClientHeight + BlockSizeY - 1) / BlockSizeY;
    cmdList->Dispatch(groupX, groupY, 1);

    cmdList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
        mDepthStencilBuffer.Get(),
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_DEPTH_WRITE
    ));

}

void SsaoApp::DrawTileMax(ID3D12GraphicsCommandList* cmdList) {

    // ------------------------------------------------------------
    // 1. Transition velocity SRV for compute read
    // ------------------------------------------------------------
    CD3DX12_RESOURCE_BARRIER toSRV =
        CD3DX12_RESOURCE_BARRIER::Transition(
            mVelocityBuffer.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,            //D3D12_RESOURCE_STATE_RENDER_TARGET, //mVelocityCurrentState,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    cmdList->ResourceBarrier(1, &toSRV);
    //mVelocityCurrentState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

    // ------------------------------------------------------------
    // 2. Transition tileMax texture for compute write (UAV)
    // ------------------------------------------------------------
    /*CD3DX12_RESOURCE_BARRIER toUAV =
        CD3DX12_RESOURCE_BARRIER::Transition(
            mTileMaxBuffer.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS //mTileMaxCurrentState,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    cmdList->ResourceBarrier(1, &toUAV);*/
    //mTileMaxCurrentState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

    // ------------------------------------------------------------
    // 3. Bind compute pipeline
    // ------------------------------------------------------------
    cmdList->SetPipelineState(mPSOs["tileMaxCS"].Get());
    cmdList->SetComputeRootSignature(mRootSignature_tileMax_mb.Get());

    // ------------------------------------------------------------
    // 4. Bind CBV, SRV, UAV
    //    Root parameter layout:
    //    0: b1 = TileMax CB
    //    1: SRV table (t2)
    //    2: UAV table (u0)
    // ------------------------------------------------------------

    // b1 = constant buffer for tile size, resolution, etc.
    cmdList->SetComputeRootConstantBufferView(
        0,
        mCurrFrameResource->tileMaxCB->Resource()->GetGPUVirtualAddress());

    // descriptor heaps
    ID3D12DescriptorHeap* heaps[] = { mSrvDescriptorHeap.Get() };
    cmdList->SetDescriptorHeaps(_countof(heaps), heaps);

    // SRV: velocity (t2)
    cmdList->SetComputeRootDescriptorTable(
        1,
        GetGpuSrv(mVelocitySrvIndex));

    // UAV: tileMax output (u0)
    cmdList->SetComputeRootDescriptorTable(
        2,
        GetGpuSrv(mTileMaxUavIndex));

    // ------------------------------------------------------------
    // 5. Calculate dispatch size
    //    TileMax output resolution = fullRes / TileSize
    //    Dispatch dimension = (width/TileSize)/8, (height/TileSize)/8
    //    Because CS uses [numthreads(8,8,1)]
    // ------------------------------------------------------------
   

   /* UINT tilesX = (mClientWidth + tileSize - 1) / tileSize;
    UINT tilesY = (mClientHeight + tileSize - 1) / tileSize;

    UINT groupsX = tilesX;
    UINT groupsY = tilesY;*/
    UINT tilesX = ceil(mClientWidth / 8); //(mClientWidth + tileSize - 1) / tileSize;
    UINT tilesY = ceil(mClientHeight / 8);//(mClientHeight + tileSize - 1) / tileSize;

    // ВАЖНО: делим на BlockSize!
    //auto BlockSizeX = 8;
    //auto BlockSizeY = 8;
    //UINT groupsX = (tilesX + BlockSizeX - 1) / BlockSizeX;
    //UINT groupsY = (tilesY + BlockSizeY - 1) / BlockSizeY;

    //cmdList->Dispatch(groupsX, groupsY, 1);
    cmdList->Dispatch(tilesX, tilesY, 1);

    // ------------------------------------------------------------
    // 6. Transition tileMax to shader-readable if next used as SRV
    // ------------------------------------------------------------
    CD3DX12_RESOURCE_BARRIER toSRV2 =
        CD3DX12_RESOURCE_BARRIER::Transition(
            mTileMaxBuffer.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, //mTileMaxCurrentState,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

    cmdList->ResourceBarrier(1, &toSRV2);
    //mTileMaxCurrentState = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;

    return;
}

void SsaoApp::DrawNeighbourMax(ID3D12GraphicsCommandList* cmdList) {
    // ------------------------------------------------------------
// 1. Transition tileMax SRV for compute read (уже сделано в конце прошлого этапа)
// ------------------------------------------------------------
    /*CD3DX12_RESOURCE_BARRIER toSRV =
        CD3DX12_RESOURCE_BARRIER::Transition(
            mTileMaxBuffer.Get(),
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    cmdList->ResourceBarrier(1, &toSRV);*/

    // ------------------------------------------------------------
    // 2. Transition neighbourMax texture for compute write (UAV)
    // ------------------------------------------------------------
    /*CD3DX12_RESOURCE_BARRIER toUAV =
        CD3DX12_RESOURCE_BARRIER::Transition(
            mNeighbourMaxBuffer.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    cmdList->ResourceBarrier(1, &toUAV);*/

    // ------------------------------------------------------------
    // 3. Bind compute pipeline
    // ------------------------------------------------------------
    cmdList->SetPipelineState(mPSOs["neighbourMaxCS"].Get());
    cmdList->SetComputeRootSignature(mRootSignature_neighbourMax_mb.Get());

    // ------------------------------------------------------------
    // 4. Bind CBV, SRV, UAV
    //    Root parameter layout:
    //    0: b2 = neighbourMax CB
    //    1: SRV table (t3)
    //    2: UAV table (u1)
    // ------------------------------------------------------------

    // b1 = constant buffer for tile size, resolution, etc.
    cmdList->SetComputeRootConstantBufferView(
        0,
        mCurrFrameResource->tileMaxCB->Resource()->GetGPUVirtualAddress());

    // descriptor heaps
    ID3D12DescriptorHeap* heaps[] = { mSrvDescriptorHeap.Get() };
    cmdList->SetDescriptorHeaps(_countof(heaps), heaps);

    // SRV: tilemax (t3)
    cmdList->SetComputeRootDescriptorTable(
        1,
        GetGpuSrv(mTileMaxSrvIndex));

    // UAV: neighbourmax output (u1)
    cmdList->SetComputeRootDescriptorTable(
        2,
        GetGpuSrv(mNeighbourMaxUavIndex));

    // ------------------------------------------------------------
    // 5. Calculate dispatch size
    //    TileMax output resolution = fullRes / TileSize
    //    Dispatch dimension = (width/TileSize)/8, (height/TileSize)/8
    //    Because CS uses [numthreads(8,8,1)]
    // ------------------------------------------------------------


    UINT tilesX = (mClientWidth + tileSize - 1) / tileSize;
    UINT tilesY = (mClientHeight + tileSize - 1) / tileSize;

    UINT groupsX = tilesX;
    UINT groupsY = tilesY;

    cmdList->Dispatch(groupsX, groupsY, 1);

    // ------------------------------------------------------------
    // 6. Transition tileMax to shader-readable if next used as SRV
    // ------------------------------------------------------------
    CD3DX12_RESOURCE_BARRIER toSRV2 =
        CD3DX12_RESOURCE_BARRIER::Transition(
            mNeighbourMaxBuffer.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, //mTileMaxCurrentState,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    cmdList->ResourceBarrier(1, &toSRV2);

    return;
}

void SsaoApp::DrawMotionBlurFinal(ID3D12GraphicsCommandList* cmdList) {

    CD3DX12_RESOURCE_BARRIER toSRV =
        CD3DX12_RESOURCE_BARRIER::Transition(
            mHdrBeforeMbBuffer.Get(),
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    cmdList->ResourceBarrier(1, &toSRV);

    cmdList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
        mDepthStencilBuffer.Get(),
        D3D12_RESOURCE_STATE_DEPTH_WRITE,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    ));

    cmdList->SetPipelineState(mPSOs["mbFinalCS"].Get());
    cmdList->SetComputeRootSignature(mRootSignature_motionBlurFinal_mb.Get());

    // descriptor heaps
    ID3D12DescriptorHeap* heaps[] = { mSrvDescriptorHeap.Get() };
    cmdList->SetDescriptorHeaps(_countof(heaps), heaps);

    cmdList->SetComputeRootConstantBufferView(
        0,
        mCurrFrameResource->MotionBlurFinalCB->Resource()->GetGPUVirtualAddress());

    cmdList->SetComputeRootDescriptorTable(1, GetGpuSrv(mHdrBeforeMbSrvIndex));
    cmdList->SetComputeRootDescriptorTable(2, GetGpuSrv(mNeighbourMaxSrvIndex));
    cmdList->SetComputeRootDescriptorTable(3, GetGpuSrv(mVelocitySrvIndex));
    cmdList->SetComputeRootDescriptorTable(4, GetGpuSrv(mCurDepthSrvIndex));
    cmdList->SetComputeRootDescriptorTable(5, GetGpuSrv(mHdrAfterMbUavIndex));

    cmdList->SetComputeRootConstantBufferView(
        6,
        mCurrFrameResource->MotionBlurCB->Resource()->GetGPUVirtualAddress());

    auto BlockSizeX = tileSize;
    auto BlockSizeY = tileSize;

    UINT groupsX = ceil(mClientWidth / BlockSizeX);//(mClientWidth + BlockSizeX - 1) / BlockSizeX;
    UINT groupsY = ceil(mClientHeight / BlockSizeY);//(mClientHeight + BlockSizeY - 1) / BlockSizeY;

    cmdList->Dispatch(groupsX, groupsY, 1);

    CD3DX12_RESOURCE_BARRIER toSRV2 =
        CD3DX12_RESOURCE_BARRIER::Transition(
            mHdrAfterMbBuffer.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, 
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

    cmdList->ResourceBarrier(1, &toSRV2);

    cmdList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
        mDepthStencilBuffer.Get(),
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_DEPTH_WRITE
    ));

    return;

}

void SsaoApp::copyDepthCurrToPrev() {

    assert(mDepthStencilBuffer.Get() != nullptr);
    assert(mDepthStencilBuffer_prev.Get() != nullptr);

    // После Present() и обновления mCurrBackBuffer
    mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
        mDepthStencilBuffer.Get(),
        D3D12_RESOURCE_STATE_DEPTH_WRITE,
        D3D12_RESOURCE_STATE_COPY_SOURCE));

    mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
        mDepthStencilBuffer_prev.Get(),
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_DEST));

    mCommandList->CopyResource(mDepthStencilBuffer_prev.Get(), mDepthStencilBuffer.Get());

    // Возврат ресурсов в привычное состояние
    mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
        mDepthStencilBuffer.Get(),
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_DEPTH_WRITE));

    mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
        mDepthStencilBuffer_prev.Get(),
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_GENERIC_READ));
}

void SsaoApp::OnMouseDown(WPARAM btnState, int x, int y)
{
    mLastMousePos.x = x;
    mLastMousePos.y = y;

    SetCapture(mhMainWnd);
}

void SsaoApp::OnMouseUp(WPARAM btnState, int x, int y)
{
    ReleaseCapture();
}

void SsaoApp::OnMouseMove(WPARAM btnState, int x, int y)
{
    if((btnState & MK_LBUTTON) != 0)
    {
		// Make each pixel correspond to a quarter of a degree.
		float dx = XMConvertToRadians(0.25f*static_cast<float>(x - mLastMousePos.x));
		float dy = XMConvertToRadians(0.25f*static_cast<float>(y - mLastMousePos.y));

		mCamera.Pitch(dy);
		mCamera.RotateY(dx);
    }

    mLastMousePos.x = x;
    mLastMousePos.y = y;
}
 
void SsaoApp::OnKeyboardInput(const GameTimer& gt)
{
	const float dt = gt.DeltaTime();

	if(GetAsyncKeyState('W') & 0x8000)
		mCamera.Walk(10.0f*dt);

	if(GetAsyncKeyState('S') & 0x8000)
		mCamera.Walk(-10.0f*dt);

	if(GetAsyncKeyState('A') & 0x8000)
		mCamera.Strafe(-10.0f*dt);

	if(GetAsyncKeyState('D') & 0x8000)
		mCamera.Strafe(10.0f*dt);

	mCamera.UpdateViewMatrix();
}
 
void SsaoApp::AnimateMaterials(const GameTimer& gt)
{
	
}

void SsaoApp::UpdateObjectCBs(const GameTimer& gt)
{
	auto currObjectCB = mCurrFrameResource->ObjectCB.get();
	for(auto& e : mAllRitems)
	{
		// Only update the cbuffer data if the constants have changed.  
		// This needs to be tracked per frame resource.
		if(e->NumFramesDirty > 0)
		{
			XMMATRIX world = XMLoadFloat4x4(&e->World);
			XMMATRIX texTransform = XMLoadFloat4x4(&e->TexTransform);

			ObjectConstants objConstants;
			XMStoreFloat4x4(&objConstants.World, XMMatrixTranspose(world));
			XMStoreFloat4x4(&objConstants.TexTransform, XMMatrixTranspose(texTransform));
			objConstants.MaterialIndex = e->Mat->MatCBIndex;

			currObjectCB->CopyData(e->ObjCBIndex, objConstants);

			// Next FrameResource need to be updated too.
			e->NumFramesDirty--;
		}
	}
}

void SsaoApp::UpdateMaterialBuffer(const GameTimer& gt)
{
	auto currMaterialBuffer = mCurrFrameResource->MaterialBuffer.get();
	for(auto& e : mMaterials)
	{
		// Only update the cbuffer data if the constants have changed.  If the cbuffer
		// data changes, it needs to be updated for each FrameResource.
		Material* mat = e.second.get();
		if(mat->NumFramesDirty > 0)
		{
			XMMATRIX matTransform = XMLoadFloat4x4(&mat->MatTransform);

			MaterialData matData;
			matData.DiffuseAlbedo = mat->DiffuseAlbedo;
			matData.FresnelR0 = mat->FresnelR0;
			matData.Roughness = mat->Roughness;
			XMStoreFloat4x4(&matData.MatTransform, XMMatrixTranspose(matTransform));
			matData.DiffuseMapIndex = mat->DiffuseSrvHeapIndex;
			matData.NormalMapIndex = mat->NormalSrvHeapIndex;

			currMaterialBuffer->CopyData(mat->MatCBIndex, matData);

			// Next FrameResource need to be updated too.
			mat->NumFramesDirty--;
		}
	}
}

void SsaoApp::UpdateShadowTransform(const GameTimer& gt)
{
    // Only the first "main" light casts a shadow.
    XMVECTOR lightDir = XMLoadFloat3(&mRotatedLightDirections[0]);
    XMVECTOR lightPos = -2.0f*mSceneBounds.Radius*lightDir;
    XMVECTOR targetPos = XMLoadFloat3(&mSceneBounds.Center);
    XMVECTOR lightUp = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);
    XMMATRIX lightView = XMMatrixLookAtLH(lightPos, targetPos, lightUp);

    XMStoreFloat3(&mLightPosW, lightPos);

    // Transform bounding sphere to light space.
    XMFLOAT3 sphereCenterLS;
    XMStoreFloat3(&sphereCenterLS, XMVector3TransformCoord(targetPos, lightView));

    // Ortho frustum in light space encloses scene.
    float l = sphereCenterLS.x - mSceneBounds.Radius;
    float b = sphereCenterLS.y - mSceneBounds.Radius;
    float n = sphereCenterLS.z - mSceneBounds.Radius;
    float r = sphereCenterLS.x + mSceneBounds.Radius;
    float t = sphereCenterLS.y + mSceneBounds.Radius;
    float f = sphereCenterLS.z + mSceneBounds.Radius;

    mLightNearZ = n;
    mLightFarZ = f;
    XMMATRIX lightProj = XMMatrixOrthographicOffCenterLH(l, r, b, t, n, f);

    // Transform NDC space [-1,+1]^2 to texture space [0,1]^2
    XMMATRIX T(
        0.5f, 0.0f, 0.0f, 0.0f,
        0.0f, -0.5f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.5f, 0.5f, 0.0f, 1.0f);

    XMMATRIX S = lightView*lightProj*T;
    XMStoreFloat4x4(&mLightView, lightView);
    XMStoreFloat4x4(&mLightProj, lightProj);
    XMStoreFloat4x4(&mShadowTransform, S);
}

void SsaoApp::UpdateMainPassCB(const GameTimer& gt)
{
	XMMATRIX view = mCamera.GetView();
	XMMATRIX proj = mCamera.GetProj();

	XMMATRIX viewProj = XMMatrixMultiply(view, proj);
	XMMATRIX invView = XMMatrixInverse(&XMMatrixDeterminant(view), view);
	XMMATRIX invProj = XMMatrixInverse(&XMMatrixDeterminant(proj), proj);
	XMMATRIX invViewProj = XMMatrixInverse(&XMMatrixDeterminant(viewProj), viewProj);

    // Transform NDC space [-1,+1]^2 to texture space [0,1]^2
    XMMATRIX T(
        0.5f, 0.0f, 0.0f, 0.0f,
        0.0f, -0.5f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.5f, 0.5f, 0.0f, 1.0f);

    XMMATRIX viewProjTex = XMMatrixMultiply(viewProj, T);
    XMMATRIX shadowTransform = XMLoadFloat4x4(&mShadowTransform);

	XMStoreFloat4x4(&mMainPassCB.View, XMMatrixTranspose(view));
	XMStoreFloat4x4(&mMainPassCB.InvView, XMMatrixTranspose(invView));
	XMStoreFloat4x4(&mMainPassCB.Proj, XMMatrixTranspose(proj));
	XMStoreFloat4x4(&mMainPassCB.InverseProjMatrix_, XMMatrixTranspose(invProj));
	XMStoreFloat4x4(&mMainPassCB.ViewProj, XMMatrixTranspose(viewProj));
	XMStoreFloat4x4(&mMainPassCB.InvViewProj, XMMatrixTranspose(invViewProj));
    XMStoreFloat4x4(&mMainPassCB.ViewProjTex, XMMatrixTranspose(viewProjTex));
    XMStoreFloat4x4(&mMainPassCB.ShadowTransform, XMMatrixTranspose(shadowTransform));
	mMainPassCB.EyePosW = mCamera.GetPosition3f();
	mMainPassCB.RenderTargetSize = XMFLOAT2((float)mClientWidth, (float)mClientHeight);
	mMainPassCB.InvRenderTargetSize = XMFLOAT2(1.0f / mClientWidth, 1.0f / mClientHeight);
	mMainPassCB.NearZ = 1.0f;
	mMainPassCB.FarZ = 1000.0f;
	mMainPassCB.TotalTime = gt.TotalTime();
	mMainPassCB.DeltaTime = gt.DeltaTime();
	mMainPassCB.AmbientLight = { 0.4f, 0.4f, 0.6f, 1.0f };
	mMainPassCB.Lights[0].Direction = mRotatedLightDirections[0];
	mMainPassCB.Lights[0].Strength = { 0.4f, 0.4f, 0.5f };
	mMainPassCB.Lights[1].Direction = mRotatedLightDirections[1];
	mMainPassCB.Lights[1].Strength = { 0.1f, 0.1f, 0.1f };
	mMainPassCB.Lights[2].Direction = mRotatedLightDirections[2];
	mMainPassCB.Lights[2].Strength = { 0.0f, 0.0f, 0.0f };
 
	auto currPassCB = mCurrFrameResource->PassCB.get();
	currPassCB->CopyData(0, mMainPassCB);
}

void SsaoApp::UpdateShadowPassCB(const GameTimer& gt)
{
    XMMATRIX view = XMLoadFloat4x4(&mLightView);
    XMMATRIX proj = XMLoadFloat4x4(&mLightProj);

    XMMATRIX viewProj = XMMatrixMultiply(view, proj);
    XMMATRIX invView = XMMatrixInverse(&XMMatrixDeterminant(view), view);
    XMMATRIX invProj = XMMatrixInverse(&XMMatrixDeterminant(proj), proj);
    XMMATRIX invViewProj = XMMatrixInverse(&XMMatrixDeterminant(viewProj), viewProj);

    UINT w = mShadowMap->Width();
    UINT h = mShadowMap->Height();

    XMStoreFloat4x4(&mShadowPassCB.View, XMMatrixTranspose(view));
    XMStoreFloat4x4(&mShadowPassCB.InvView, XMMatrixTranspose(invView));
    XMStoreFloat4x4(&mShadowPassCB.Proj, XMMatrixTranspose(proj));
    XMStoreFloat4x4(&mShadowPassCB.InverseProjMatrix_, XMMatrixTranspose(invProj));
    XMStoreFloat4x4(&mShadowPassCB.ViewProj, XMMatrixTranspose(viewProj));
    XMStoreFloat4x4(&mShadowPassCB.InvViewProj, XMMatrixTranspose(invViewProj));
    mShadowPassCB.EyePosW = mLightPosW;
    mShadowPassCB.RenderTargetSize = XMFLOAT2((float)w, (float)h);
    mShadowPassCB.InvRenderTargetSize = XMFLOAT2(1.0f / w, 1.0f / h);
    mShadowPassCB.NearZ = mLightNearZ;
    mShadowPassCB.FarZ = mLightFarZ;

    auto currPassCB = mCurrFrameResource->PassCB.get();
    currPassCB->CopyData(1, mShadowPassCB);
}

void SsaoApp::UpdateSsaoCB(const GameTimer& gt)
{
    SsaoConstants ssaoCB;

    XMMATRIX P = mCamera.GetProj();

    // Transform NDC space [-1,+1]^2 to texture space [0,1]^2
    XMMATRIX T(
        0.5f, 0.0f, 0.0f, 0.0f,
        0.0f, -0.5f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.5f, 0.5f, 0.0f, 1.0f);

    ssaoCB.Proj    = mMainPassCB.Proj;
    ssaoCB.InverseProjMatrix_ = mMainPassCB.InverseProjMatrix_;
    XMStoreFloat4x4(&ssaoCB.ProjTex, XMMatrixTranspose(P*T));

    mSsao->GetOffsetVectors(ssaoCB.OffsetVectors);

    auto blurWeights = mSsao->CalcGaussWeights(2.5f);
    ssaoCB.BlurWeights[0] = XMFLOAT4(&blurWeights[0]);
    ssaoCB.BlurWeights[1] = XMFLOAT4(&blurWeights[4]);
    ssaoCB.BlurWeights[2] = XMFLOAT4(&blurWeights[8]);

    ssaoCB.InvRenderTargetSize = XMFLOAT2(1.0f / mSsao->SsaoMapWidth(), 1.0f / mSsao->SsaoMapHeight());

    // Coordinates given in view space.
    ssaoCB.OcclusionRadius = 0.5f;
    ssaoCB.OcclusionFadeStart = 0.2f;
    ssaoCB.OcclusionFadeEnd = 1.0f;
    ssaoCB.SurfaceEpsilon = 0.05f;
 
    auto currSsaoCB = mCurrFrameResource->SsaoCB.get();
    currSsaoCB->CopyData(0, ssaoCB);
}

void SsaoApp::LoadTextures()
{
	std::vector<std::string> texNames = 
	{
		"bricksDiffuseMap",
		"bricksNormalMap",
		"tileDiffuseMap",
		"tileNormalMap",
		"defaultDiffuseMap",
		"defaultNormalMap",
		"skyCubeMap"
	};
	
    std::vector<std::wstring> texFilenames =
    {
        L"../../Textures/bricks2.dds",
        L"../../Textures/bricks2_nmap.dds",
        L"../../Textures/tile.dds",
        L"../../Textures/tile_nmap.dds",
        L"../../Textures/white1x1.dds",
        L"../../Textures/default_nmap.dds",
        L"../../Textures/sunsetcube1024.dds"
    };
	
	for(int i = 0; i < (int)texNames.size(); ++i)
	{
		auto texMap = std::make_unique<Texture>();
		texMap->Name = texNames[i];
		texMap->Filename = texFilenames[i];
		ThrowIfFailed(DirectX::CreateDDSTextureFromFile12(md3dDevice.Get(),
			mCommandList.Get(), texMap->Filename.c_str(),
			texMap->Resource, texMap->UploadHeap));
			
		mTextures[texMap->Name] = std::move(texMap);
	}		
}

void SsaoApp::BuildRootSignature()
{
	CD3DX12_DESCRIPTOR_RANGE texTable0;
	texTable0.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 3, 0, 0);

	CD3DX12_DESCRIPTOR_RANGE texTable1;
	texTable1.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 10, 3, 0);

    // Root parameter can be a table, root descriptor or root constants.
    CD3DX12_ROOT_PARAMETER slotRootParameter[5];

	// Perfomance TIP: Order from most frequent to least frequent.
    slotRootParameter[0].InitAsConstantBufferView(0);
    slotRootParameter[1].InitAsConstantBufferView(1);
    slotRootParameter[2].InitAsShaderResourceView(0, 1);
	slotRootParameter[3].InitAsDescriptorTable(1, &texTable0, D3D12_SHADER_VISIBILITY_PIXEL);
	slotRootParameter[4].InitAsDescriptorTable(1, &texTable1, D3D12_SHADER_VISIBILITY_PIXEL);


	auto staticSamplers = GetStaticSamplers();

    // A root signature is an array of root parameters.
	CD3DX12_ROOT_SIGNATURE_DESC rootSigDesc(5, slotRootParameter,
		(UINT)staticSamplers.size(), staticSamplers.data(),
		D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

    // create a root signature with a single slot which points to a descriptor range consisting of a single constant buffer
    ComPtr<ID3DBlob> serializedRootSig = nullptr;
    ComPtr<ID3DBlob> errorBlob = nullptr;
    HRESULT hr = D3D12SerializeRootSignature(&rootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1,
        serializedRootSig.GetAddressOf(), errorBlob.GetAddressOf());

    if(errorBlob != nullptr)
    {
        ::OutputDebugStringA((char*)errorBlob->GetBufferPointer());
    }
    ThrowIfFailed(hr);

    ThrowIfFailed(md3dDevice->CreateRootSignature(
		0,
        serializedRootSig->GetBufferPointer(),
        serializedRootSig->GetBufferSize(),
        IID_PPV_ARGS(mRootSignature.GetAddressOf())));
}

void SsaoApp::BuildRootSignature_mb() {
    // velocity stage

   // версия для графики
   /* CD3DX12_DESCRIPTOR_RANGE depth0;
    depth0.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);

    CD3DX12_DESCRIPTOR_RANGE depth1;
    depth1.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1);

    CD3DX12_ROOT_PARAMETER slotRootParams[3];

    slotRootParams[0].InitAsConstantBufferView(0); // matrices cb (b0)
    slotRootParams[1].InitAsDescriptorTable(1, &depth0);
    slotRootParams[2].InitAsDescriptorTable(1, &depth1);

    CD3DX12_STATIC_SAMPLER_DESC linearSampler(
        0, D3D12_FILTER_MIN_MAG_MIP_LINEAR);

    CD3DX12_ROOT_SIGNATURE_DESC rootSigDesc(
        3, slotRootParams,
        1, &linearSampler,
        D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

    ComPtr<ID3DBlob> serializedRootSig = nullptr;
    ComPtr<ID3DBlob> errorBlob = nullptr;
    D3D12SerializeRootSignature(
        &rootSigDesc,
        D3D_ROOT_SIGNATURE_VERSION_1,
        &serializedRootSig, &errorBlob);

    md3dDevice->CreateRootSignature(
        0,
        serializedRootSig->GetBufferPointer(),
        serializedRootSig->GetBufferSize(),
        IID_PPV_ARGS(mRootSignature_velocity_mb.GetAddressOf()));
        */

    // компьют версия
    CD3DX12_DESCRIPTOR_RANGE depth0;
    depth0.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0); // t0
    CD3DX12_DESCRIPTOR_RANGE depth1;
    depth1.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1); // t1
    CD3DX12_DESCRIPTOR_RANGE velocityUav;
    velocityUav.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 2); // u2

    CD3DX12_ROOT_PARAMETER rootParamsVelocity[4];
    rootParamsVelocity[0].InitAsConstantBufferView(0);      // b0
    rootParamsVelocity[1].InitAsDescriptorTable(1, &depth0); // t0
    rootParamsVelocity[2].InitAsDescriptorTable(1, &depth1); // t1
    rootParamsVelocity[3].InitAsDescriptorTable(1, &velocityUav); // u2

    CD3DX12_ROOT_SIGNATURE_DESC rootSigDescVelocity(
        4, rootParamsVelocity,
        0, nullptr,
        D3D12_ROOT_SIGNATURE_FLAG_NONE
    );

    ComPtr<ID3DBlob> serializedRootSig = nullptr;
    ComPtr<ID3DBlob> errorBlob = nullptr;
    D3D12SerializeRootSignature(
        &rootSigDescVelocity,
        D3D_ROOT_SIGNATURE_VERSION_1,
        &serializedRootSig, &errorBlob);

    md3dDevice->CreateRootSignature(
        0,
        serializedRootSig->GetBufferPointer(),
        serializedRootSig->GetBufferSize(),
        IID_PPV_ARGS(mRootSignature_velocity_cs_mb.GetAddressOf()));

    // -------------------------------- tilemax stage -----------------------------

    // это для графич шейдера
    /*CD3DX12_DESCRIPTOR_RANGE srvRange;
    srvRange.Init(
        D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        1,      // one SRV
        2       // t2
    );

    CD3DX12_ROOT_PARAMETER params[2];

    // ---- b1: TileMaxCB ----
    params[0].InitAsConstantBufferView(1);  // (shader register b1)

    // ---- SRV table: t2 ----
    params[1].InitAsDescriptorTable(1, &srvRange);

    // ---- Sampler s0 ----
    CD3DX12_STATIC_SAMPLER_DESC sampler(
        0, // register s0
        D3D12_FILTER_MIN_MAG_MIP_LINEAR,
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP
    );

    CD3DX12_ROOT_SIGNATURE_DESC rootSigDescTM(
        _countof(params),
        params,
        1, &sampler,
        D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
    );

    ComPtr<ID3DBlob> sigBlob;
    ComPtr<ID3DBlob> errorBlobTM;

    HRESULT hr = D3D12SerializeRootSignature(
        &rootSigDescTM,
        D3D_ROOT_SIGNATURE_VERSION_1,
        &sigBlob,
        &errorBlobTM
    );

    if (FAILED(hr))
    {
        if (errorBlobTM)
        {
            OutputDebugStringA((char*)errorBlobTM->GetBufferPointer());
        }
        ThrowIfFailed(hr);
    }

    ThrowIfFailed(md3dDevice->CreateRootSignature(
        0,
        sigBlob->GetBufferPointer(),
        sigBlob->GetBufferSize(),
        IID_PPV_ARGS(&mRootSignature_tileMax_mb)
    ));
    */

    // это для компьют

    // ---- SRV (velocity buffer): t2 ----
    CD3DX12_DESCRIPTOR_RANGE srvRange;
    srvRange.Init(
        D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        1,  // count
        2   // register t2
    );

    // ---- UAV (tileMax output): u0 ----
    CD3DX12_DESCRIPTOR_RANGE uavRange;
    uavRange.Init(
        D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
        1,
        0   // register u0
    );

    CD3DX12_ROOT_PARAMETER params[3];

    // b1 = TileMax constant buffer
    params[0].InitAsConstantBufferView(1);

    // SRV table: t2
    params[1].InitAsDescriptorTable(1, &srvRange);

    // UAV table: u0
    params[2].InitAsDescriptorTable(1, &uavRange);

    // Sampler s0 (linear clamp)
    CD3DX12_STATIC_SAMPLER_DESC sampler(
        0, // s0
        D3D12_FILTER_MIN_MAG_MIP_LINEAR,
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP
    );

    // NOTE: compute signature must NOT use IA flags!
    CD3DX12_ROOT_SIGNATURE_DESC rootSigDescTM(
        _countof(params), params,
        0, nullptr,// 1, &sampler,
        D3D12_ROOT_SIGNATURE_FLAG_NONE
    );

    ComPtr<ID3DBlob> sigBlob;
    ComPtr<ID3DBlob> errorBlobTM;

    ThrowIfFailed(D3D12SerializeRootSignature(
        &rootSigDescTM,
        D3D_ROOT_SIGNATURE_VERSION_1,
        &sigBlob,
        &errorBlobTM));

    ThrowIfFailed(md3dDevice->CreateRootSignature(
        0,
        sigBlob->GetBufferPointer(),
        sigBlob->GetBufferSize(),
        IID_PPV_ARGS(&mRootSignature_tileMax_mb)));

    // то же самое для neighbour компьюта
    // сначала надо написать шейдер и определиться с регистрами
    
        // ---- SRV (tilemax buffer): t3 ----
    CD3DX12_DESCRIPTOR_RANGE srvRangeNM;
    srvRangeNM.Init(
        D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        1,  // count
        3   // register t3
    );

    // ---- UAV (neighbourMax output): u1 ----
    CD3DX12_DESCRIPTOR_RANGE uavRangeNM;
    uavRangeNM.Init(
        D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
        1,
        1   // register u1
    );

    CD3DX12_ROOT_PARAMETER paramsNM[3];

    // b2 = NeighbourMax constant buffer
    paramsNM[0].InitAsConstantBufferView(2);

    // SRV table: t3
    paramsNM[1].InitAsDescriptorTable(1, &srvRangeNM);

    // UAV table: u1
    paramsNM[2].InitAsDescriptorTable(1, &uavRangeNM);

    // сэмплер у нас уже написан
    // Sampler s0 (linear clamp)
/*    CD3DX12_STATIC_SAMPLER_DESC sampler(
        0, // s0
        D3D12_FILTER_MIN_MAG_MIP_LINEAR,
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP
    );*/

    // NOTE: compute signature must NOT use IA flags!
    CD3DX12_ROOT_SIGNATURE_DESC rootSigDescNM(
        _countof(paramsNM), paramsNM,
        0, nullptr, // 1, &sampler,
        D3D12_ROOT_SIGNATURE_FLAG_NONE
    );

    ComPtr<ID3DBlob> sigBlobNM;
    ComPtr<ID3DBlob> errorBlobNM;

    ThrowIfFailed(D3D12SerializeRootSignature(
        &rootSigDescNM,
        D3D_ROOT_SIGNATURE_VERSION_1,
        &sigBlobNM,
        &errorBlobNM));

    ThrowIfFailed(md3dDevice->CreateRootSignature(
        0,
        sigBlobNM->GetBufferPointer(),
        sigBlobNM->GetBufferSize(),
        IID_PPV_ARGS(&mRootSignature_neighbourMax_mb)));

    /// ----------------- mb Final stage -------------------------------

    // mbfinalroot
    CD3DX12_DESCRIPTOR_RANGE hdrSourceSrv;
    hdrSourceSrv.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 4); // 

    CD3DX12_DESCRIPTOR_RANGE neighbourSrv;
    neighbourSrv.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 5); // t5

    CD3DX12_DESCRIPTOR_RANGE velocitySrvMbFin;
    velocitySrvMbFin.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 6); //  

    CD3DX12_DESCRIPTOR_RANGE depthSrv;
    depthSrv.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 7); // t7 

    // ------ UAV table -------
    // u3: HdrTarget
    CD3DX12_DESCRIPTOR_RANGE uavRangeMbFinal;
    uavRangeMbFinal.Init(
        D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
        1,
        3
    );

    CD3DX12_ROOT_PARAMETER rootParamsMbFinal[7];

    rootParamsMbFinal[0].InitAsConstantBufferView(3);              // b3
    rootParamsMbFinal[1].InitAsDescriptorTable(1, &hdrSourceSrv);
    rootParamsMbFinal[2].InitAsDescriptorTable(1, &neighbourSrv);  
    rootParamsMbFinal[3].InitAsDescriptorTable(1, &velocitySrvMbFin); 
    rootParamsMbFinal[4].InitAsDescriptorTable(1, &depthSrv); 
    rootParamsMbFinal[5].InitAsDescriptorTable(1, &uavRangeMbFinal); 
    rootParamsMbFinal[6].InitAsConstantBufferView(0);              // b0

    // ----------- Root signature desc -----------
    
    CD3DX12_STATIC_SAMPLER_DESC staticSamplersFinal[2];

    CD3DX12_STATIC_SAMPLER_DESC samplerPoint(
        1, // s1
        D3D12_FILTER_MIN_MAG_MIP_POINT,
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP
    );
    staticSamplersFinal[0] = sampler;
    staticSamplersFinal[1] = samplerPoint;
    
    CD3DX12_ROOT_SIGNATURE_DESC rootSigDescMbFinal(
        _countof(rootParamsMbFinal), rootParamsMbFinal,
        _countof(staticSamplersFinal), staticSamplersFinal,
        D3D12_ROOT_SIGNATURE_FLAG_NONE
    );

    ComPtr<ID3DBlob> sigBlobMbFinal;
    ComPtr<ID3DBlob> errorBlobMbFinal;

    ThrowIfFailed(D3D12SerializeRootSignature(
        &rootSigDescMbFinal,
        D3D_ROOT_SIGNATURE_VERSION_1,
        &sigBlobMbFinal,
        &errorBlobMbFinal
    ));

    ThrowIfFailed(md3dDevice->CreateRootSignature(
        0,
        sigBlobMbFinal->GetBufferPointer(),
        sigBlobMbFinal->GetBufferSize(),
        IID_PPV_ARGS(&mRootSignature_motionBlurFinal_mb)
    ));

}

void SsaoApp::BuildSsaoRootSignature()
{
    CD3DX12_DESCRIPTOR_RANGE texTable0;
    texTable0.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 0, 0);

    CD3DX12_DESCRIPTOR_RANGE texTable1;
    texTable1.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 2, 0);

    // Root parameter can be a table, root descriptor or root constants.
    CD3DX12_ROOT_PARAMETER slotRootParameter[4];

    // Perfomance TIP: Order from most frequent to least frequent.
    slotRootParameter[0].InitAsConstantBufferView(0);
    slotRootParameter[1].InitAsConstants(1, 1);
    slotRootParameter[2].InitAsDescriptorTable(1, &texTable0, D3D12_SHADER_VISIBILITY_PIXEL);
    slotRootParameter[3].InitAsDescriptorTable(1, &texTable1, D3D12_SHADER_VISIBILITY_PIXEL);

    const CD3DX12_STATIC_SAMPLER_DESC pointClamp(
        0, // shaderRegister
        D3D12_FILTER_MIN_MAG_MIP_POINT, // filter
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP,  // addressU
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP,  // addressV
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP); // addressW

    const CD3DX12_STATIC_SAMPLER_DESC linearClamp(
        1, // shaderRegister
        D3D12_FILTER_MIN_MAG_MIP_LINEAR, // filter
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP,  // addressU
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP,  // addressV
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP); // addressW

    const CD3DX12_STATIC_SAMPLER_DESC depthMapSam(
        2, // shaderRegister
        D3D12_FILTER_MIN_MAG_MIP_LINEAR, // filter
        D3D12_TEXTURE_ADDRESS_MODE_BORDER,  // addressU
        D3D12_TEXTURE_ADDRESS_MODE_BORDER,  // addressV
        D3D12_TEXTURE_ADDRESS_MODE_BORDER,  // addressW
        0.0f,
        0,
        D3D12_COMPARISON_FUNC_LESS_EQUAL,
        D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE); 

    const CD3DX12_STATIC_SAMPLER_DESC linearWrap(
        3, // shaderRegister
        D3D12_FILTER_MIN_MAG_MIP_LINEAR, // filter
        D3D12_TEXTURE_ADDRESS_MODE_WRAP,  // addressU
        D3D12_TEXTURE_ADDRESS_MODE_WRAP,  // addressV
        D3D12_TEXTURE_ADDRESS_MODE_WRAP); // addressW

    std::array<CD3DX12_STATIC_SAMPLER_DESC, 4> staticSamplers =
    {
        pointClamp, linearClamp, depthMapSam, linearWrap
    };

    // A root signature is an array of root parameters.
    CD3DX12_ROOT_SIGNATURE_DESC rootSigDesc(4, slotRootParameter,
        (UINT)staticSamplers.size(), staticSamplers.data(),
        D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

    // create a root signature with a single slot which points to a descriptor range consisting of a single constant buffer
    ComPtr<ID3DBlob> serializedRootSig = nullptr;
    ComPtr<ID3DBlob> errorBlob = nullptr;
    HRESULT hr = D3D12SerializeRootSignature(&rootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1,
        serializedRootSig.GetAddressOf(), errorBlob.GetAddressOf());

    if(errorBlob != nullptr)
    {
        ::OutputDebugStringA((char*)errorBlob->GetBufferPointer());
    }
    ThrowIfFailed(hr);

    ThrowIfFailed(md3dDevice->CreateRootSignature(
        0,
        serializedRootSig->GetBufferPointer(),
        serializedRootSig->GetBufferSize(),
        IID_PPV_ARGS(mSsaoRootSignature.GetAddressOf())));
}

void SsaoApp::BuildDescriptorHeaps()
{
	//
	// Create the SRV heap.
	//
	D3D12_DESCRIPTOR_HEAP_DESC srvHeapDesc = {};
	srvHeapDesc.NumDescriptors = 60;
	srvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	srvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	ThrowIfFailed(md3dDevice->CreateDescriptorHeap(&srvHeapDesc, IID_PPV_ARGS(&mSrvDescriptorHeap)));

	//
	// Fill out the heap with actual descriptors.
	//
	CD3DX12_CPU_DESCRIPTOR_HANDLE hDescriptor(mSrvDescriptorHeap->GetCPUDescriptorHandleForHeapStart());

	std::vector<ComPtr<ID3D12Resource>> tex2DList = 
	{
		mTextures["bricksDiffuseMap"]->Resource,
		mTextures["bricksNormalMap"]->Resource,
		mTextures["tileDiffuseMap"]->Resource,
		mTextures["tileNormalMap"]->Resource,
		mTextures["defaultDiffuseMap"]->Resource,
		mTextures["defaultNormalMap"]->Resource
	};
	
	auto skyCubeMap = mTextures["skyCubeMap"]->Resource;

	D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
	srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
	srvDesc.Texture2D.MostDetailedMip = 0;
	srvDesc.Texture2D.ResourceMinLODClamp = 0.0f;
	
	for(UINT i = 0; i < (UINT)tex2DList.size(); ++i)
	{
		srvDesc.Format = tex2DList[i]->GetDesc().Format;
		srvDesc.Texture2D.MipLevels = tex2DList[i]->GetDesc().MipLevels;
		md3dDevice->CreateShaderResourceView(tex2DList[i].Get(), &srvDesc, hDescriptor);

		// next descriptor
		hDescriptor.Offset(1, mCbvSrvUavDescriptorSize);
	}
	
	srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
	srvDesc.TextureCube.MostDetailedMip = 0;
	srvDesc.TextureCube.MipLevels = skyCubeMap->GetDesc().MipLevels;
	srvDesc.TextureCube.ResourceMinLODClamp = 0.0f;
	srvDesc.Format = skyCubeMap->GetDesc().Format;
	md3dDevice->CreateShaderResourceView(skyCubeMap.Get(), &srvDesc, hDescriptor);
	
	mSkyTexHeapIndex = (UINT)tex2DList.size();
    mShadowMapHeapIndex = mSkyTexHeapIndex + 1;
    mSsaoHeapIndexStart = mShadowMapHeapIndex + 1;
    mSsaoAmbientMapIndex = mSsaoHeapIndexStart + 3;
    mNullCubeSrvIndex = mSsaoHeapIndexStart + 5;
    mNullTexSrvIndex1 = mNullCubeSrvIndex + 1;
    mNullTexSrvIndex2 = mNullTexSrvIndex1 + 1;

    // MB -----------------------------------------------
    mPrevDepthSrvIndex = mNullTexSrvIndex2 + 1;

    mTileMaxSrvIndex = mPrevDepthSrvIndex + 1;
    mTileMaxUavIndex = mTileMaxSrvIndex + 1;

    mNeighbourMaxUavIndex = mTileMaxUavIndex + 1;

    mVelocityUavIndex = mNeighbourMaxUavIndex + 1;

    mHdrAfterMbSrvIndex = mVelocityUavIndex + 1;
    mHdrAfterMbUavIndex = mHdrAfterMbSrvIndex + 1;

    // ОНИ ДОЛЖНЫ БЫТЬ ПОДРЯД! root signature такая
    mCurDepthSrvIndex = mHdrAfterMbUavIndex + 1; // t0
    mVelocitySrvIndex = mCurDepthSrvIndex + 1; // t2
    mHdrBeforeMbSrvIndex = mVelocitySrvIndex + 1; // t4
    mNeighbourMaxSrvIndex = mHdrBeforeMbSrvIndex + 1; // t5

    mVelocityRtvIndex = SwapChainBufferCount + 3; // см размер кучи rtv в crateRtv.....()
    mHdrBeforeMbRtvIndex = SwapChainBufferCount + 4;

    // MB: второй буфер глубины
    CD3DX12_CPU_DESCRIPTOR_HANDLE hPrevDepthSrv = GetCpuSrv(mPrevDepthSrvIndex);

    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc_depth_prev = {};
    srvDesc_depth_prev.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc_depth_prev.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDesc_depth_prev.Format = DXGI_FORMAT_R24_UNORM_X8_TYPELESS; // если depth буфер
    srvDesc_depth_prev.Texture2D.MostDetailedMip = 0;
    srvDesc_depth_prev.Texture2D.MipLevels = 1;
    srvDesc_depth_prev.Texture2D.ResourceMinLODClamp = 0.0f;

    md3dDevice->CreateShaderResourceView(mDepthStencilBuffer_prev.Get(), &srvDesc_depth_prev, hPrevDepthSrv);

    // srv для имеющегося буфера глубины

    CD3DX12_CPU_DESCRIPTOR_HANDLE hCurDepthSrv = GetCpuSrv(mCurDepthSrvIndex);

    // Описатель SRV
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDescCurDepth = {};
    srvDescCurDepth.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDescCurDepth.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDescCurDepth.Format = DXGI_FORMAT_R24_UNORM_X8_TYPELESS; // формат depth buffer
    srvDescCurDepth.Texture2D.MostDetailedMip = 0;
    srvDescCurDepth.Texture2D.MipLevels = 1;
    srvDescCurDepth.Texture2D.ResourceMinLODClamp = 0.0f;

    md3dDevice->CreateShaderResourceView(
        mDepthStencilBuffer.Get(),   // убедись, что это тот ресурс, что пишет d3dApp
        &srvDescCurDepth,
        hCurDepthSrv
    );


    // буфер скоростей : srv + rtv (+ uav потому что перешли на компьют)

    auto velSrvCpu = GetCpuSrv(mVelocitySrvIndex);

    D3D12_SHADER_RESOURCE_VIEW_DESC srvDescVelocitySrv = {};
    srvDescVelocitySrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDescVelocitySrv.Format = DXGI_FORMAT_R16G16_FLOAT;
    srvDescVelocitySrv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDescVelocitySrv.Texture2D.MostDetailedMip = 0;
    srvDescVelocitySrv.Texture2D.MipLevels = 1;
    srvDescVelocitySrv.Texture2D.ResourceMinLODClamp = 0.0f;

    md3dDevice->CreateShaderResourceView(
        mVelocityBuffer.Get(),
        &srvDescVelocitySrv,
        velSrvCpu);

    auto rtvHandle = GetRtv(mVelocityRtvIndex);

    D3D12_RENDER_TARGET_VIEW_DESC rtvDesc = {};
    rtvDesc.Format = DXGI_FORMAT_R16G16_FLOAT;
    rtvDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;

    md3dDevice->CreateRenderTargetView(mVelocityBuffer.Get(), &rtvDesc, rtvHandle);

    auto velUavCpu = GetCpuSrv(mVelocityUavIndex);

    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDescVelocity = {};
    uavDescVelocity.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    uavDescVelocity.Format = DXGI_FORMAT_R16G16_FLOAT;
    md3dDevice->CreateUnorderedAccessView(mVelocityBuffer.Get(), nullptr, &uavDescVelocity, velUavCpu);

    // буфер TileMax: srv + uav

    // SRV для TileMax (чтобы его потом читать)
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDescTile = {};
    srvDescTile.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDescTile.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDescTile.Format = DXGI_FORMAT_R16G16_FLOAT;
    srvDescTile.Texture2D.MostDetailedMip = 0;
    srvDescTile.Texture2D.MipLevels = 1;
    srvDescTile.Texture2D.ResourceMinLODClamp = 0.0f;

    md3dDevice->CreateShaderResourceView(mTileMaxBuffer.Get(), &srvDescTile, GetCpuSrv(mTileMaxSrvIndex));

    // UAV для TileMax (чтобы compute мог писать)
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDescTile = {};
    uavDescTile.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    uavDescTile.Format = DXGI_FORMAT_R16G16_FLOAT;
    uavDescTile.Texture2D.MipSlice = 0;
    uavDescTile.Texture2D.PlaneSlice = 0;

    md3dDevice->CreateUnorderedAccessView(mTileMaxBuffer.Get(), nullptr, &uavDescTile, GetCpuSrv(mTileMaxUavIndex));

    // то же самое для Neighbour Max: srv + uav

    // SRV для NeighbourMax (чтобы его потом читать)
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDescNeighbour = {};
    srvDescNeighbour.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDescNeighbour.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDescNeighbour.Format = DXGI_FORMAT_R16G16_FLOAT;
    srvDescNeighbour.Texture2D.MostDetailedMip = 0;
    srvDescNeighbour.Texture2D.MipLevels = 1;
    srvDescNeighbour.Texture2D.ResourceMinLODClamp = 0.0f;

    md3dDevice->CreateShaderResourceView(mNeighbourMaxBuffer.Get(), &srvDescNeighbour, GetCpuSrv(mNeighbourMaxSrvIndex));

    // UAV для NeighbourMax (чтобы compute мог писать)
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDescNeighbour = {};
    uavDescNeighbour.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    uavDescNeighbour.Format = DXGI_FORMAT_R16G16_FLOAT;
    uavDescNeighbour.Texture2D.MipSlice = 0;
    uavDescNeighbour.Texture2D.PlaneSlice = 0;

    md3dDevice->CreateUnorderedAccessView(mNeighbourMaxBuffer.Get(), nullptr, &uavDescNeighbour, GetCpuSrv(mNeighbourMaxUavIndex));

    // srv для hdrBefor

    auto hdrBeforeSrvCpu = GetCpuSrv(mHdrBeforeMbSrvIndex);

    D3D12_SHADER_RESOURCE_VIEW_DESC srvDescHdrBeforSrv = {};
    srvDescHdrBeforSrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDescHdrBeforSrv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    srvDescHdrBeforSrv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDescHdrBeforSrv.Texture2D.MostDetailedMip = 0;
    srvDescHdrBeforSrv.Texture2D.MipLevels = 1;
    srvDescHdrBeforSrv.Texture2D.ResourceMinLODClamp = 0.0f;

    md3dDevice->CreateShaderResourceView(
        mHdrBeforeMbBuffer.Get(),
        &srvDescHdrBeforSrv,
        hdrBeforeSrvCpu);

    // rtv для hdrBefor

    auto rtvHandleHdrBefore = GetRtv(mHdrBeforeMbRtvIndex);

    D3D12_RENDER_TARGET_VIEW_DESC rtvDescHdrBefore = {};
    rtvDescHdrBefore.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    rtvDescHdrBefore.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;

    md3dDevice->CreateRenderTargetView(mHdrBeforeMbBuffer.Get(), &rtvDescHdrBefore, rtvHandleHdrBefore);

    // srv для hdrAfter
    auto hdrAfterSrvCpu = GetCpuSrv(mHdrAfterMbSrvIndex);

    D3D12_SHADER_RESOURCE_VIEW_DESC srvDeschdrAfterSrv = {};
    srvDeschdrAfterSrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDeschdrAfterSrv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    srvDeschdrAfterSrv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDeschdrAfterSrv.Texture2D.MostDetailedMip = 0;
    srvDeschdrAfterSrv.Texture2D.MipLevels = 1;
    srvDeschdrAfterSrv.Texture2D.ResourceMinLODClamp = 0.0f;

    md3dDevice->CreateShaderResourceView(
        mHdrAfterMbBuffer.Get(),
        &srvDeschdrAfterSrv,
        hdrAfterSrvCpu);


    // uav для hdrAfter

    auto hdrAfterUavCpu = GetCpuSrv(mHdrAfterMbUavIndex);

    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDescHdrAfter = {};
    uavDescHdrAfter.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    uavDescHdrAfter.Format = DXGI_FORMAT_R8G8B8A8_UNORM;

    md3dDevice->CreateUnorderedAccessView(mHdrAfterMbBuffer.Get(), nullptr, &uavDescHdrAfter, hdrAfterUavCpu);

    //////////// ----------------------------------------

    auto nullSrv = GetCpuSrv(mNullCubeSrvIndex);
    mNullSrv = GetGpuSrv(mNullCubeSrvIndex);

    md3dDevice->CreateShaderResourceView(nullptr, &srvDesc, nullSrv);
    nullSrv.Offset(1, mCbvSrvUavDescriptorSize);

    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    srvDesc.Texture2D.MostDetailedMip = 0;
    srvDesc.Texture2D.MipLevels = 1;
    srvDesc.Texture2D.ResourceMinLODClamp = 0.0f;
    md3dDevice->CreateShaderResourceView(nullptr, &srvDesc, nullSrv);

    nullSrv.Offset(1, mCbvSrvUavDescriptorSize);
    md3dDevice->CreateShaderResourceView(nullptr, &srvDesc, nullSrv);

    mShadowMap->BuildDescriptors(
        GetCpuSrv(mShadowMapHeapIndex),
        GetGpuSrv(mShadowMapHeapIndex),
        GetDsv(1));
    

    mSsao->BuildDescriptors(
        mDepthStencilBuffer.Get(),
        GetCpuSrv(mSsaoHeapIndexStart),
        GetGpuSrv(mSsaoHeapIndexStart),
        GetRtv(SwapChainBufferCount),
        mCbvSrvUavDescriptorSize,
        mRtvDescriptorSize);
}

class DxcBlobAsID3DBlob : public ID3DBlob
{
public:
    DxcBlobAsID3DBlob(ComPtr<IDxcBlob> blob) : m_blob(blob) {}

    LPVOID STDMETHODCALLTYPE GetBufferPointer() override {
        return m_blob->GetBufferPointer();
    }

    SIZE_T STDMETHODCALLTYPE GetBufferSize() override {
        return m_blob->GetBufferSize();
    }

    // Reference counting:
    ULONG STDMETHODCALLTYPE AddRef() override { return ++m_ref; }
    ULONG STDMETHODCALLTYPE Release() override {
        if (--m_ref == 0) { delete this; return 0; }
        return m_ref;
    }
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppv) override {
        if (riid == __uuidof(ID3DBlob)) {
            *ppv = this;
            AddRef();
            return S_OK;
        }
        *ppv = nullptr;
        return E_NOINTERFACE;
    }

private:
    std::atomic_uint m_ref = 1;
    ComPtr<IDxcBlob> m_blob;
};

void SsaoApp::BuildShadersAndInputLayout()
{
	const D3D_SHADER_MACRO alphaTestDefines[] =
	{
		"ALPHA_TEST", "1",
		NULL, NULL
	};

	mShaders["standardVS"] = d3dUtil::CompileShader(L"Shaders\\Default.hlsl", nullptr, "VS", "vs_5_1");
	mShaders["opaquePS"] = d3dUtil::CompileShader(L"Shaders\\Default.hlsl", nullptr, "PS", "ps_5_1");

    mShaders["shadowVS"] = d3dUtil::CompileShader(L"Shaders\\Shadows.hlsl", nullptr, "VS", "vs_5_1");
    mShaders["shadowOpaquePS"] = d3dUtil::CompileShader(L"Shaders\\Shadows.hlsl", nullptr, "PS", "ps_5_1");
    mShaders["shadowAlphaTestedPS"] = d3dUtil::CompileShader(L"Shaders\\Shadows.hlsl", alphaTestDefines, "PS", "ps_5_1");
	
    mShaders["debugVS"] = d3dUtil::CompileShader(L"Shaders\\ShadowDebug.hlsl", nullptr, "VS", "vs_5_1");
    mShaders["debugPS"] = d3dUtil::CompileShader(L"Shaders\\ShadowDebug.hlsl", nullptr, "PS", "ps_5_1");

    mShaders["drawNormalsVS"] = d3dUtil::CompileShader(L"Shaders\\DrawNormals.hlsl", nullptr, "VS", "vs_5_1");
    mShaders["drawNormalsPS"] = d3dUtil::CompileShader(L"Shaders\\DrawNormals.hlsl", nullptr, "PS", "ps_5_1");

    mShaders["ssaoVS"] = d3dUtil::CompileShader(L"Shaders\\Ssao.hlsl", nullptr, "VS", "vs_5_1");
    mShaders["ssaoPS"] = d3dUtil::CompileShader(L"Shaders\\Ssao.hlsl", nullptr, "PS", "ps_5_1");

    mShaders["ssaoBlurVS"] = d3dUtil::CompileShader(L"Shaders\\SsaoBlur.hlsl", nullptr, "VS", "vs_5_1");
    mShaders["ssaoBlurPS"] = d3dUtil::CompileShader(L"Shaders\\SsaoBlur.hlsl", nullptr, "PS", "ps_5_1");

	mShaders["skyVS"] = d3dUtil::CompileShader(L"Shaders\\Sky.hlsl", nullptr, "VS", "vs_5_1");
	mShaders["skyPS"] = d3dUtil::CompileShader(L"Shaders\\Sky.hlsl", nullptr, "PS", "ps_5_1");

    // mb часть
    mShaders["velocityVS"] =
        d3dUtil::CompileShader(L"Shaders\\MotionBlur.hlsl", nullptr, "VS", "vs_5_1");

    mShaders["velocityPS"] =
        d3dUtil::CompileShader(L"Shaders\\MotionBlur.hlsl", nullptr, "PSMain", "ps_5_1");

    auto velocityShaderBlob =
        CompileShaderSM6(
            L"Shaders\\MotionBlur.hlsl",
            L"CSMain",
            L"cs_6_0"
        );
    mShaders["velocityCS"] = new DxcBlobAsID3DBlob(velocityShaderBlob);

    //mShaders["tileMaxCS"] =
    //    d3dUtil::CompileShader(L"Shaders\\MotionBlur.hlsl", nullptr, "CS_TileMax", "cs_5_1");

    /*D3D_SHADER_MACRO tileMaxDefines[] = {
    { "BlockSizeX", "8" },
    { "BlockSizeY", "8" },
    { nullptr, nullptr }
    }; */

    auto tileMaxShaderBlob =
        CompileShaderSM6(
            L"Shaders\\MotionBlur.hlsl",
            L"CS_TileMax",
            L"cs_6_0"
        );
    mShaders["tileMaxCS"] = new DxcBlobAsID3DBlob(tileMaxShaderBlob); //d3dUtil::CompileShader(L"Shaders\\MotionBlur.hlsl", nullptr, "CS_TileMax", "cs_6_0");
    
    auto NMshaderBlob =
        CompileShaderSM6(
            L"Shaders\\MotionBlur.hlsl",
            L"CS_NeighbourMax",
            L"cs_6_0"
        );

    mShaders["neighbourMaxCS"] = new DxcBlobAsID3DBlob(NMshaderBlob);
        //d3dUtil::CompileShader(L"Shaders\\MotionBlur.hlsl", nullptr, "CS_NeighbourMax", "cs_6_0");

    auto mbFinalshaderBlob =
        CompileShaderSM6(
            L"Shaders\\MotionBlur.hlsl",
            L"CS_motionBlurFinal",
            L"cs_6_0"
        );

    mShaders["mbFinalCS"] = new DxcBlobAsID3DBlob(mbFinalshaderBlob);

    mInputLayout =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
		{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 24, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
		{ "TANGENT", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 32, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
    };
}

#include <iostream>

ComPtr<IDxcBlob> SsaoApp::CompileShaderSM6(
    const std::wstring& filename,
    const std::wstring& entryPoint,
    const std::wstring& target)
{
    using namespace Microsoft::WRL;

    ComPtr<IDxcUtils> utils;
    ComPtr<IDxcCompiler3> compiler;
    ComPtr<IDxcIncludeHandler> includeHandler;

    // Создаём DXC объекты
    DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&utils));
    DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&compiler));
    utils->CreateDefaultIncludeHandler(&includeHandler);

    // Загружаем файл шейдера
    ComPtr<IDxcBlobEncoding> sourceBlob;
    utils->LoadFile(filename.c_str(), nullptr, &sourceBlob);

    DxcBuffer sourceBuffer;
    sourceBuffer.Ptr = sourceBlob->GetBufferPointer();
    sourceBuffer.Size = sourceBlob->GetBufferSize();
    sourceBuffer.Encoding = DXC_CP_UTF8;

    // Аргументы компиляции SM6
    std::wstring entry = L"-E";
    std::wstring targetFlag = L"-T";

    LPCWSTR args[] = {
        filename.c_str(),
        L"-Zi",                   // debug info
        L"-O3",                   // оптимизация
        entry.c_str(),            // "-E"
        entryPoint.c_str(),       // имя функции
        targetFlag.c_str(),       // "-T"
        target.c_str(),           // например "cs_6_0"
    };

    ComPtr<IDxcResult> result;
    HRESULT hr = compiler->Compile(
        &sourceBuffer,
        args,
        _countof(args),
        includeHandler.Get(),
        IID_PPV_ARGS(&result));


    if (FAILED(hr))
    {
        throw std::runtime_error("Failed to compile shader (DXC compile call failed)");
    }

    // Проверяем ошибки
    ComPtr<IDxcBlobUtf8> errors;
    result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errors), nullptr);
    if (errors && errors->GetStringLength() > 0)
    {
        std::cout << errors->GetStringPointer() << std::endl;
        OutputDebugStringA(errors->GetStringPointer());
    }

    // Получаем скомпилированный blob
    ComPtr<IDxcBlob> shaderBlob;
    result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&shaderBlob), nullptr);

    return shaderBlob;
}

void SsaoApp::BuildShapeGeometry()
{
    GeometryGenerator geoGen;
	GeometryGenerator::MeshData box = geoGen.CreateBox(1.0f, 1.0f, 1.0f, 3);
	GeometryGenerator::MeshData grid = geoGen.CreateGrid(20.0f, 30.0f, 60, 40);
	GeometryGenerator::MeshData sphere = geoGen.CreateSphere(0.5f, 20, 20);
	GeometryGenerator::MeshData cylinder = geoGen.CreateCylinder(0.5f, 0.3f, 3.0f, 20, 20);
    GeometryGenerator::MeshData quad = geoGen.CreateQuad(0.0f, 0.0f, 1.0f, 1.0f, 0.0f);
    
	//
	// We are concatenating all the geometry into one big vertex/index buffer.  So
	// define the regions in the buffer each submesh covers.
	//

	// Cache the vertex offsets to each object in the concatenated vertex buffer.
	UINT boxVertexOffset = 0;
	UINT gridVertexOffset = (UINT)box.Vertices.size();
	UINT sphereVertexOffset = gridVertexOffset + (UINT)grid.Vertices.size();
	UINT cylinderVertexOffset = sphereVertexOffset + (UINT)sphere.Vertices.size();
    UINT quadVertexOffset = cylinderVertexOffset + (UINT)cylinder.Vertices.size();

	// Cache the starting index for each object in the concatenated index buffer.
	UINT boxIndexOffset = 0;
	UINT gridIndexOffset = (UINT)box.Indices32.size();
	UINT sphereIndexOffset = gridIndexOffset + (UINT)grid.Indices32.size();
	UINT cylinderIndexOffset = sphereIndexOffset + (UINT)sphere.Indices32.size();
    UINT quadIndexOffset = cylinderIndexOffset + (UINT)cylinder.Indices32.size();

	SubmeshGeometry boxSubmesh;
	boxSubmesh.IndexCount = (UINT)box.Indices32.size();
	boxSubmesh.StartIndexLocation = boxIndexOffset;
	boxSubmesh.BaseVertexLocation = boxVertexOffset;

	SubmeshGeometry gridSubmesh;
	gridSubmesh.IndexCount = (UINT)grid.Indices32.size();
	gridSubmesh.StartIndexLocation = gridIndexOffset;
	gridSubmesh.BaseVertexLocation = gridVertexOffset;

	SubmeshGeometry sphereSubmesh;
	sphereSubmesh.IndexCount = (UINT)sphere.Indices32.size();
	sphereSubmesh.StartIndexLocation = sphereIndexOffset;
	sphereSubmesh.BaseVertexLocation = sphereVertexOffset;

	SubmeshGeometry cylinderSubmesh;
	cylinderSubmesh.IndexCount = (UINT)cylinder.Indices32.size();
	cylinderSubmesh.StartIndexLocation = cylinderIndexOffset;
	cylinderSubmesh.BaseVertexLocation = cylinderVertexOffset;

    SubmeshGeometry quadSubmesh;
    quadSubmesh.IndexCount = (UINT)quad.Indices32.size();
    quadSubmesh.StartIndexLocation = quadIndexOffset;
    quadSubmesh.BaseVertexLocation = quadVertexOffset;

	//
	// Extract the vertex elements we are interested in and pack the
	// vertices of all the meshes into one vertex buffer.
	//

	auto totalVertexCount =
		box.Vertices.size() +
		grid.Vertices.size() +
		sphere.Vertices.size() +
		cylinder.Vertices.size() + 
        quad.Vertices.size();

	std::vector<Vertex> vertices(totalVertexCount);

	UINT k = 0;
	for(size_t i = 0; i < box.Vertices.size(); ++i, ++k)
	{
		vertices[k].Pos = box.Vertices[i].Position;
		vertices[k].Normal = box.Vertices[i].Normal;
		vertices[k].TexC = box.Vertices[i].TexC;
		vertices[k].TangentU = box.Vertices[i].TangentU;
	}

	for(size_t i = 0; i < grid.Vertices.size(); ++i, ++k)
	{
		vertices[k].Pos = grid.Vertices[i].Position;
		vertices[k].Normal = grid.Vertices[i].Normal;
		vertices[k].TexC = grid.Vertices[i].TexC;
		vertices[k].TangentU = grid.Vertices[i].TangentU;
	}

	for(size_t i = 0; i < sphere.Vertices.size(); ++i, ++k)
	{
		vertices[k].Pos = sphere.Vertices[i].Position;
		vertices[k].Normal = sphere.Vertices[i].Normal;
		vertices[k].TexC = sphere.Vertices[i].TexC;
		vertices[k].TangentU = sphere.Vertices[i].TangentU;
	}

	for(size_t i = 0; i < cylinder.Vertices.size(); ++i, ++k)
	{
		vertices[k].Pos = cylinder.Vertices[i].Position;
		vertices[k].Normal = cylinder.Vertices[i].Normal;
		vertices[k].TexC = cylinder.Vertices[i].TexC;
		vertices[k].TangentU = cylinder.Vertices[i].TangentU;
	}

    for(int i = 0; i < quad.Vertices.size(); ++i, ++k)
    {
        vertices[k].Pos = quad.Vertices[i].Position;
        vertices[k].Normal = quad.Vertices[i].Normal;
        vertices[k].TexC = quad.Vertices[i].TexC;
        vertices[k].TangentU = quad.Vertices[i].TangentU;
    }

	std::vector<std::uint16_t> indices;
	indices.insert(indices.end(), std::begin(box.GetIndices16()), std::end(box.GetIndices16()));
	indices.insert(indices.end(), std::begin(grid.GetIndices16()), std::end(grid.GetIndices16()));
	indices.insert(indices.end(), std::begin(sphere.GetIndices16()), std::end(sphere.GetIndices16()));
	indices.insert(indices.end(), std::begin(cylinder.GetIndices16()), std::end(cylinder.GetIndices16()));
    indices.insert(indices.end(), std::begin(quad.GetIndices16()), std::end(quad.GetIndices16()));

    const UINT vbByteSize = (UINT)vertices.size() * sizeof(Vertex);
    const UINT ibByteSize = (UINT)indices.size()  * sizeof(std::uint16_t);

	auto geo = std::make_unique<MeshGeometry>();
	geo->Name = "shapeGeo";

	ThrowIfFailed(D3DCreateBlob(vbByteSize, &geo->VertexBufferCPU));
	CopyMemory(geo->VertexBufferCPU->GetBufferPointer(), vertices.data(), vbByteSize);

	ThrowIfFailed(D3DCreateBlob(ibByteSize, &geo->IndexBufferCPU));
	CopyMemory(geo->IndexBufferCPU->GetBufferPointer(), indices.data(), ibByteSize);

	geo->VertexBufferGPU = d3dUtil::CreateDefaultBuffer(md3dDevice.Get(),
		mCommandList.Get(), vertices.data(), vbByteSize, geo->VertexBufferUploader);

	geo->IndexBufferGPU = d3dUtil::CreateDefaultBuffer(md3dDevice.Get(),
		mCommandList.Get(), indices.data(), ibByteSize, geo->IndexBufferUploader);

	geo->VertexByteStride = sizeof(Vertex);
	geo->VertexBufferByteSize = vbByteSize;
	geo->IndexFormat = DXGI_FORMAT_R16_UINT;
	geo->IndexBufferByteSize = ibByteSize;

	geo->DrawArgs["box"] = boxSubmesh;
	geo->DrawArgs["grid"] = gridSubmesh;
	geo->DrawArgs["sphere"] = sphereSubmesh;
	geo->DrawArgs["cylinder"] = cylinderSubmesh;
    geo->DrawArgs["quad"] = quadSubmesh;

	mGeometries[geo->Name] = std::move(geo);
}

void SsaoApp::BuildSkullGeometry()
{
    std::ifstream fin("Models/skull.txt");

    if (!fin)
    {
        MessageBox(0, L"Models/skull.txt not found.", 0, 0);
        return;
    }

    UINT vcount = 0;
    UINT tcount = 0;
    std::string ignore;

    fin >> ignore >> vcount;
    fin >> ignore >> tcount;
    fin >> ignore >> ignore >> ignore >> ignore;

    XMFLOAT3 vMinf3(+MathHelper::Infinity, +MathHelper::Infinity, +MathHelper::Infinity);
    XMFLOAT3 vMaxf3(-MathHelper::Infinity, -MathHelper::Infinity, -MathHelper::Infinity);

    XMVECTOR vMin = XMLoadFloat3(&vMinf3);
    XMVECTOR vMax = XMLoadFloat3(&vMaxf3);

    std::vector<Vertex> vertices(vcount);
    for (UINT i = 0; i < vcount; ++i)
    {
        fin >> vertices[i].Pos.x >> vertices[i].Pos.y >> vertices[i].Pos.z;
        fin >> vertices[i].Normal.x >> vertices[i].Normal.y >> vertices[i].Normal.z;

        vertices[i].TexC = { 0.0f, 0.0f };

        XMVECTOR P = XMLoadFloat3(&vertices[i].Pos);

        XMVECTOR N = XMLoadFloat3(&vertices[i].Normal);

        // Generate a tangent vector so normal mapping works.  We aren't applying
        // a texture map to the skull, so we just need any tangent vector so that
        // the math works out to give us the original interpolated vertex normal.
        XMVECTOR up = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);
        if (fabsf(XMVectorGetX(XMVector3Dot(N, up))) < 1.0f - 0.001f)
        {
            XMVECTOR T = XMVector3Normalize(XMVector3Cross(up, N));
            XMStoreFloat3(&vertices[i].TangentU, T);
        }
        else
        {
            up = XMVectorSet(0.0f, 0.0f, 1.0f, 0.0f);
            XMVECTOR T = XMVector3Normalize(XMVector3Cross(N, up));
            XMStoreFloat3(&vertices[i].TangentU, T);
        }


        vMin = XMVectorMin(vMin, P);
        vMax = XMVectorMax(vMax, P);
    }

    BoundingBox bounds;
    XMStoreFloat3(&bounds.Center, 0.5f*(vMin + vMax));
    XMStoreFloat3(&bounds.Extents, 0.5f*(vMax - vMin));

    fin >> ignore;
    fin >> ignore;
    fin >> ignore;

    std::vector<std::int32_t> indices(3 * tcount);
    for (UINT i = 0; i < tcount; ++i)
    {
        fin >> indices[i * 3 + 0] >> indices[i * 3 + 1] >> indices[i * 3 + 2];
    }

    fin.close();

    //
    // Pack the indices of all the meshes into one index buffer.
    //

    const UINT vbByteSize = (UINT)vertices.size() * sizeof(Vertex);

    const UINT ibByteSize = (UINT)indices.size() * sizeof(std::int32_t);

    auto geo = std::make_unique<MeshGeometry>();
    geo->Name = "skullGeo";

    ThrowIfFailed(D3DCreateBlob(vbByteSize, &geo->VertexBufferCPU));
    CopyMemory(geo->VertexBufferCPU->GetBufferPointer(), vertices.data(), vbByteSize);

    ThrowIfFailed(D3DCreateBlob(ibByteSize, &geo->IndexBufferCPU));
    CopyMemory(geo->IndexBufferCPU->GetBufferPointer(), indices.data(), ibByteSize);

    geo->VertexBufferGPU = d3dUtil::CreateDefaultBuffer(md3dDevice.Get(),
        mCommandList.Get(), vertices.data(), vbByteSize, geo->VertexBufferUploader);

    geo->IndexBufferGPU = d3dUtil::CreateDefaultBuffer(md3dDevice.Get(),
        mCommandList.Get(), indices.data(), ibByteSize, geo->IndexBufferUploader);

    geo->VertexByteStride = sizeof(Vertex);
    geo->VertexBufferByteSize = vbByteSize;
    geo->IndexFormat = DXGI_FORMAT_R32_UINT;
    geo->IndexBufferByteSize = ibByteSize;

    SubmeshGeometry submesh;
    submesh.IndexCount = (UINT)indices.size();
    submesh.StartIndexLocation = 0;
    submesh.BaseVertexLocation = 0;
    submesh.Bounds = bounds;

    geo->DrawArgs["skull"] = submesh;

    mGeometries[geo->Name] = std::move(geo);
}

void SsaoApp::BuildPSOs()
{
    D3D12_GRAPHICS_PIPELINE_STATE_DESC basePsoDesc;

	
    ZeroMemory(&basePsoDesc, sizeof(D3D12_GRAPHICS_PIPELINE_STATE_DESC));
    basePsoDesc.InputLayout = { mInputLayout.data(), (UINT)mInputLayout.size() };
    basePsoDesc.pRootSignature = mRootSignature.Get();
    basePsoDesc.VS =
	{ 
		reinterpret_cast<BYTE*>(mShaders["standardVS"]->GetBufferPointer()), 
		mShaders["standardVS"]->GetBufferSize()
	};
    basePsoDesc.PS =
	{ 
		reinterpret_cast<BYTE*>(mShaders["opaquePS"]->GetBufferPointer()),
		mShaders["opaquePS"]->GetBufferSize()
	};
    basePsoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    basePsoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
    basePsoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
    basePsoDesc.SampleMask = UINT_MAX;
    basePsoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    basePsoDesc.NumRenderTargets = 1;
    basePsoDesc.RTVFormats[0] = mBackBufferFormat;
    basePsoDesc.SampleDesc.Count = m4xMsaaState ? 4 : 1;
    basePsoDesc.SampleDesc.Quality = m4xMsaaState ? (m4xMsaaQuality - 1) : 0;
    basePsoDesc.DSVFormat = mDepthStencilFormat;

    //
    // PSO for opaque objects.
    //

    D3D12_GRAPHICS_PIPELINE_STATE_DESC opaquePsoDesc = basePsoDesc;
    opaquePsoDesc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_EQUAL;
    opaquePsoDesc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
    ThrowIfFailed(md3dDevice->CreateGraphicsPipelineState(&opaquePsoDesc, IID_PPV_ARGS(&mPSOs["opaque"])));

    //
    // PSO for shadow map pass.
    //
    D3D12_GRAPHICS_PIPELINE_STATE_DESC smapPsoDesc = basePsoDesc;
    smapPsoDesc.RasterizerState.DepthBias = 100000;
    smapPsoDesc.RasterizerState.DepthBiasClamp = 0.0f;
    smapPsoDesc.RasterizerState.SlopeScaledDepthBias = 1.0f;
    smapPsoDesc.pRootSignature = mRootSignature.Get();
    smapPsoDesc.VS =
    {
        reinterpret_cast<BYTE*>(mShaders["shadowVS"]->GetBufferPointer()),
        mShaders["shadowVS"]->GetBufferSize()
    };
    smapPsoDesc.PS =
    {
        reinterpret_cast<BYTE*>(mShaders["shadowOpaquePS"]->GetBufferPointer()),
        mShaders["shadowOpaquePS"]->GetBufferSize()
    };
    
    // Shadow map pass does not have a render target.
    smapPsoDesc.RTVFormats[0] = DXGI_FORMAT_UNKNOWN;
    smapPsoDesc.NumRenderTargets = 0;
    ThrowIfFailed(md3dDevice->CreateGraphicsPipelineState(&smapPsoDesc, IID_PPV_ARGS(&mPSOs["shadow_opaque"])));

    //
    // PSO for debug layer.
    //
    D3D12_GRAPHICS_PIPELINE_STATE_DESC debugPsoDesc = basePsoDesc;
    debugPsoDesc.pRootSignature = mRootSignature.Get();
    debugPsoDesc.VS =
    {
        reinterpret_cast<BYTE*>(mShaders["debugVS"]->GetBufferPointer()),
        mShaders["debugVS"]->GetBufferSize()
    };
    debugPsoDesc.PS =
    {
        reinterpret_cast<BYTE*>(mShaders["debugPS"]->GetBufferPointer()),
        mShaders["debugPS"]->GetBufferSize()
    };
    ThrowIfFailed(md3dDevice->CreateGraphicsPipelineState(&debugPsoDesc, IID_PPV_ARGS(&mPSOs["debug"])));

    //
    // PSO for drawing normals.
    //
    D3D12_GRAPHICS_PIPELINE_STATE_DESC drawNormalsPsoDesc = basePsoDesc;
    drawNormalsPsoDesc.VS =
    {
        reinterpret_cast<BYTE*>(mShaders["drawNormalsVS"]->GetBufferPointer()),
        mShaders["drawNormalsVS"]->GetBufferSize()
    };
    drawNormalsPsoDesc.PS =
    {
        reinterpret_cast<BYTE*>(mShaders["drawNormalsPS"]->GetBufferPointer()),
        mShaders["drawNormalsPS"]->GetBufferSize()
    };
    drawNormalsPsoDesc.RTVFormats[0] = Ssao::NormalMapFormat;
    drawNormalsPsoDesc.SampleDesc.Count = 1;
    drawNormalsPsoDesc.SampleDesc.Quality = 0;
    drawNormalsPsoDesc.DSVFormat = mDepthStencilFormat;
    ThrowIfFailed(md3dDevice->CreateGraphicsPipelineState(&drawNormalsPsoDesc, IID_PPV_ARGS(&mPSOs["drawNormals"])));

    // TODO: motion blur all this lines
    //
    // PSO for SSAO.
    //
    D3D12_GRAPHICS_PIPELINE_STATE_DESC ssaoPsoDesc = basePsoDesc;
    ssaoPsoDesc.InputLayout = { nullptr, 0 };
    ssaoPsoDesc.pRootSignature = mSsaoRootSignature.Get();
    ssaoPsoDesc.VS =
    {
        reinterpret_cast<BYTE*>(mShaders["ssaoVS"]->GetBufferPointer()),
        mShaders["ssaoVS"]->GetBufferSize()
    };
    ssaoPsoDesc.PS =
    {
        reinterpret_cast<BYTE*>(mShaders["ssaoPS"]->GetBufferPointer()),
        mShaders["ssaoPS"]->GetBufferSize()
    };

    // SSAO effect does not need the depth buffer.
    ssaoPsoDesc.DepthStencilState.DepthEnable = false;
    ssaoPsoDesc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
    ssaoPsoDesc.RTVFormats[0] = Ssao::AmbientMapFormat;
    ssaoPsoDesc.SampleDesc.Count = 1;
    ssaoPsoDesc.SampleDesc.Quality = 0;
    ssaoPsoDesc.DSVFormat = DXGI_FORMAT_UNKNOWN;
    ThrowIfFailed(md3dDevice->CreateGraphicsPipelineState(&ssaoPsoDesc, IID_PPV_ARGS(&mPSOs["ssao"])));

    //
    // PSO for SSAO blur.
    //
    D3D12_GRAPHICS_PIPELINE_STATE_DESC ssaoBlurPsoDesc = ssaoPsoDesc;
    ssaoBlurPsoDesc.VS =
    {
        reinterpret_cast<BYTE*>(mShaders["ssaoBlurVS"]->GetBufferPointer()),
        mShaders["ssaoBlurVS"]->GetBufferSize()
    };
    ssaoBlurPsoDesc.PS =
    {
        reinterpret_cast<BYTE*>(mShaders["ssaoBlurPS"]->GetBufferPointer()),
        mShaders["ssaoBlurPS"]->GetBufferSize()
    };
    ThrowIfFailed(md3dDevice->CreateGraphicsPipelineState(&ssaoBlurPsoDesc, IID_PPV_ARGS(&mPSOs["ssaoBlur"])));

    // MB часть
    BuildPSO_mb(basePsoDesc);

	//
	// PSO for sky.
	//
	D3D12_GRAPHICS_PIPELINE_STATE_DESC skyPsoDesc = basePsoDesc;

	// The camera is inside the sky sphere, so just turn off culling.
	skyPsoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;

	// Make sure the depth function is LESS_EQUAL and not just LESS.  
	// Otherwise, the normalized depth values at z = 1 (NDC) will 
	// fail the depth test if the depth buffer was cleared to 1.
	skyPsoDesc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
	skyPsoDesc.pRootSignature = mRootSignature.Get();
	skyPsoDesc.VS =
	{
		reinterpret_cast<BYTE*>(mShaders["skyVS"]->GetBufferPointer()),
		mShaders["skyVS"]->GetBufferSize()
	};
	skyPsoDesc.PS =
	{
		reinterpret_cast<BYTE*>(mShaders["skyPS"]->GetBufferPointer()),
		mShaders["skyPS"]->GetBufferSize()
	};
	ThrowIfFailed(md3dDevice->CreateGraphicsPipelineState(&skyPsoDesc, IID_PPV_ARGS(&mPSOs["sky"])));

}

void SsaoApp::BuildPSO_mb(D3D12_GRAPHICS_PIPELINE_STATE_DESC basePsoDesc) {
    // Velocity stage
    
    // версия для графического шейдера
    /*
    // Базовая конфигурация — копируем basePsoDesc как делает SSAO
    D3D12_GRAPHICS_PIPELINE_STATE_DESC velocityPsoDesc = basePsoDesc;

    // Полноэкранный пасс — без вершинных буферов
    velocityPsoDesc.InputLayout = { nullptr, 0 };

    // Наша рут сигнатура velocity-pass
    velocityPsoDesc.pRootSignature = mRootSignature_velocity_mb.Get();

    // Подключаем шейдеры
    velocityPsoDesc.VS =
    {
        reinterpret_cast<BYTE*>(mShaders["velocityVS"]->GetBufferPointer()),
        mShaders["velocityVS"]->GetBufferSize()
    };

    velocityPsoDesc.PS =
    {
        reinterpret_cast<BYTE*>(mShaders["velocityPS"]->GetBufferPointer()),
        mShaders["velocityPS"]->GetBufferSize()
    };

    velocityPsoDesc.DepthStencilState.DepthEnable = false;
    velocityPsoDesc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
    velocityPsoDesc.DSVFormat = DXGI_FORMAT_UNKNOWN;

    velocityPsoDesc.RTVFormats[0] = DXGI_FORMAT_R16G16_FLOAT;

    // Single-sample render target
    velocityPsoDesc.SampleDesc.Count = 1;
    velocityPsoDesc.SampleDesc.Quality = 0;

    // Создаём PSO
    ThrowIfFailed(
        md3dDevice->CreateGraphicsPipelineState(
            &velocityPsoDesc, IID_PPV_ARGS(&mPSOs["velocity"]))
    ); */

    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDescVelocity = {};
    psoDescVelocity.pRootSignature = mRootSignature_velocity_cs_mb.Get();
    psoDescVelocity.CS = { mShaders["velocityCS"]->GetBufferPointer(), mShaders["velocityCS"]->GetBufferSize()};
    psoDescVelocity.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;

    ThrowIfFailed(md3dDevice->CreateComputePipelineState(
        &psoDescVelocity,
        IID_PPV_ARGS(&mPSOs["velocityCS"])
    ));

    // ------------------- TileMax stage --------------------------------

    //
// TileMax Compute PSO
//

    D3D12_COMPUTE_PIPELINE_STATE_DESC tilemaxCSDesc = {};
    tilemaxCSDesc.pRootSignature = mRootSignature_tileMax_mb.Get();


    // CS шейдер
    tilemaxCSDesc.CS = {
        reinterpret_cast<BYTE*>(mShaders["tileMaxCS"]->GetBufferPointer()),
        mShaders["tileMaxCS"]->GetBufferSize()
    };

    // Флаги обычно 0
    tilemaxCSDesc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;

    ThrowIfFailed(md3dDevice->CreateComputePipelineState(
        &tilemaxCSDesc,
        IID_PPV_ARGS(&mPSOs["tileMaxCS"])
    ));

    /*D3D12_GRAPHICS_PIPELINE_STATE_DESC tilemaxPsoDesc = basePsoDesc;

    tilemaxPsoDesc.InputLayout = { nullptr, 0 };
    tilemaxPsoDesc.pRootSignature = mRootSignature_tileMax_mb.Get();

    tilemaxPsoDesc.VS = {
        reinterpret_cast<BYTE*>(mShaders["velocityVS"]->GetBufferPointer()),
        mShaders["velocityVS"]->GetBufferSize()
    };

    tilemaxPsoDesc.PS = {
        reinterpret_cast<BYTE*>(mShaders["tileMaxPS"]->GetBufferPointer()),
        mShaders["tileMaxPS"]->GetBufferSize()
    };

    // no depth
    tilemaxPsoDesc.DepthStencilState.DepthEnable = false;
    tilemaxPsoDesc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
    tilemaxPsoDesc.DSVFormat = DXGI_FORMAT_UNKNOWN;

    tilemaxPsoDesc.NumRenderTargets = 1;
    tilemaxPsoDesc.RTVFormats[0] = DXGI_FORMAT_R16G16_FLOAT;

    tilemaxPsoDesc.SampleDesc.Count = 1;
    tilemaxPsoDesc.SampleDesc.Quality = 0;

    tilemaxPsoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    tilemaxPsoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);

    ThrowIfFailed(md3dDevice->CreateGraphicsPipelineState(
        &tilemaxPsoDesc, IID_PPV_ARGS(&mPSOs["tilemax"])
    ));*/

    // neighbour max Compute PSO
    // neigbourMaxCS

    D3D12_COMPUTE_PIPELINE_STATE_DESC neighbourmaxCSDesc = {};
    neighbourmaxCSDesc.pRootSignature = mRootSignature_neighbourMax_mb.Get();

    // CS шейдер
    neighbourmaxCSDesc.CS = {
        reinterpret_cast<BYTE*>(mShaders["neighbourMaxCS"]->GetBufferPointer()),
        mShaders["neighbourMaxCS"]->GetBufferSize()
    };

    // Флаги обычно 0
    neighbourmaxCSDesc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;

    ThrowIfFailed(md3dDevice->CreateComputePipelineState(
        &neighbourmaxCSDesc,
        IID_PPV_ARGS(&mPSOs["neighbourMaxCS"])
    ));

    // mbFinalCS

    D3D12_COMPUTE_PIPELINE_STATE_DESC mbFinalCSDesc = {};
    mbFinalCSDesc.pRootSignature = mRootSignature_motionBlurFinal_mb.Get();

    // CS шейдер
    mbFinalCSDesc.CS = {
        reinterpret_cast<BYTE*>(mShaders["mbFinalCS"]->GetBufferPointer()),
        mShaders["mbFinalCS"]->GetBufferSize()
    };

    // Флаги обычно 0
    mbFinalCSDesc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;

    ThrowIfFailed(md3dDevice->CreateComputePipelineState(
        &mbFinalCSDesc,
        IID_PPV_ARGS(&mPSOs["mbFinalCS"])
    ));
}

void SsaoApp::BuildFrameResources()
{
    for(int i = 0; i < gNumFrameResources; ++i)
    {
        mFrameResources.push_back(std::make_unique<FrameResource>(md3dDevice.Get(),
            2, (UINT)mAllRitems.size(), (UINT)mMaterials.size()));
    }
}

void SsaoApp::BuildMaterials()
{
    auto bricks0 = std::make_unique<Material>();
    bricks0->Name = "bricks0";
    bricks0->MatCBIndex = 0;
    bricks0->DiffuseSrvHeapIndex = 0;
    bricks0->NormalSrvHeapIndex = 1;
    bricks0->DiffuseAlbedo = XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f);
    bricks0->FresnelR0 = XMFLOAT3(0.1f, 0.1f, 0.1f);
    bricks0->Roughness = 0.3f;

    auto tile0 = std::make_unique<Material>();
    tile0->Name = "tile0";
    tile0->MatCBIndex = 2;
    tile0->DiffuseSrvHeapIndex = 2;
    tile0->NormalSrvHeapIndex = 3;
    tile0->DiffuseAlbedo = XMFLOAT4(0.9f, 0.9f, 0.9f, 1.0f);
    tile0->FresnelR0 = XMFLOAT3(0.2f, 0.2f, 0.2f);
    tile0->Roughness = 0.1f;

    auto mirror0 = std::make_unique<Material>();
    mirror0->Name = "mirror0";
    mirror0->MatCBIndex = 3;
    mirror0->DiffuseSrvHeapIndex = 4;
    mirror0->NormalSrvHeapIndex = 5;
    mirror0->DiffuseAlbedo = XMFLOAT4(0.0f, 0.0f, 0.0f, 1.0f);
    mirror0->FresnelR0 = XMFLOAT3(0.98f, 0.97f, 0.95f);
    mirror0->Roughness = 0.1f;

    auto skullMat = std::make_unique<Material>();
    skullMat->Name = "skullMat";
    skullMat->MatCBIndex = 3;
    skullMat->DiffuseSrvHeapIndex = 4;
    skullMat->NormalSrvHeapIndex = 5;
    skullMat->DiffuseAlbedo = XMFLOAT4(0.3f, 0.3f, 0.3f, 1.0f);
    skullMat->FresnelR0 = XMFLOAT3(0.6f, 0.6f, 0.6f);
    skullMat->Roughness = 0.2f;

    auto sky = std::make_unique<Material>();
    sky->Name = "sky";
    sky->MatCBIndex = 4;
    sky->DiffuseSrvHeapIndex = 6;
    sky->NormalSrvHeapIndex = 7;
    sky->DiffuseAlbedo = XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f);
    sky->FresnelR0 = XMFLOAT3(0.1f, 0.1f, 0.1f);
    sky->Roughness = 1.0f;

    mMaterials["bricks0"] = std::move(bricks0);
    mMaterials["tile0"] = std::move(tile0);
    mMaterials["mirror0"] = std::move(mirror0);
    mMaterials["skullMat"] = std::move(skullMat);
    mMaterials["sky"] = std::move(sky);
}

void SsaoApp::BuildRenderItems()
{
	auto skyRitem = std::make_unique<RenderItem>();
	XMStoreFloat4x4(&skyRitem->World, XMMatrixScaling(5000.0f, 5000.0f, 5000.0f));
	skyRitem->TexTransform = MathHelper::Identity4x4();
	skyRitem->ObjCBIndex = 0;
	skyRitem->Mat = mMaterials["sky"].get();
	skyRitem->Geo = mGeometries["shapeGeo"].get();
	skyRitem->PrimitiveType = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
	skyRitem->IndexCount = skyRitem->Geo->DrawArgs["sphere"].IndexCount;
	skyRitem->StartIndexLocation = skyRitem->Geo->DrawArgs["sphere"].StartIndexLocation;
	skyRitem->BaseVertexLocation = skyRitem->Geo->DrawArgs["sphere"].BaseVertexLocation;

	mRitemLayer[(int)RenderLayer::Sky].push_back(skyRitem.get());
	mAllRitems.push_back(std::move(skyRitem));
    
    auto quadRitem = std::make_unique<RenderItem>();
    quadRitem->World = MathHelper::Identity4x4();
    quadRitem->TexTransform = MathHelper::Identity4x4();
    quadRitem->ObjCBIndex = 1;
    quadRitem->Mat = mMaterials["bricks0"].get();
    quadRitem->Geo = mGeometries["shapeGeo"].get();
    quadRitem->PrimitiveType = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    quadRitem->IndexCount = quadRitem->Geo->DrawArgs["quad"].IndexCount;
    quadRitem->StartIndexLocation = quadRitem->Geo->DrawArgs["quad"].StartIndexLocation;
    quadRitem->BaseVertexLocation = quadRitem->Geo->DrawArgs["quad"].BaseVertexLocation;

    mRitemLayer[(int)RenderLayer::Debug].push_back(quadRitem.get());
    mAllRitems.push_back(std::move(quadRitem));
    
	auto boxRitem = std::make_unique<RenderItem>();
	XMStoreFloat4x4(&boxRitem->World, XMMatrixScaling(2.0f, 1.0f, 2.0f)*XMMatrixTranslation(0.0f, 0.5f, 0.0f));
	XMStoreFloat4x4(&boxRitem->TexTransform, XMMatrixScaling(1.0f, 0.5f, 1.0f));
	boxRitem->ObjCBIndex = 2;
	boxRitem->Mat = mMaterials["bricks0"].get();
	boxRitem->Geo = mGeometries["shapeGeo"].get();
	boxRitem->PrimitiveType = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
	boxRitem->IndexCount = boxRitem->Geo->DrawArgs["box"].IndexCount;
	boxRitem->StartIndexLocation = boxRitem->Geo->DrawArgs["box"].StartIndexLocation;
	boxRitem->BaseVertexLocation = boxRitem->Geo->DrawArgs["box"].BaseVertexLocation;

	mRitemLayer[(int)RenderLayer::Opaque].push_back(boxRitem.get());
	mAllRitems.push_back(std::move(boxRitem));

    auto skullRitem = std::make_unique<RenderItem>();
    XMStoreFloat4x4(&skullRitem->World, XMMatrixScaling(0.4f, 0.4f, 0.4f)*XMMatrixTranslation(0.0f, 1.0f, 0.0f));
    skullRitem->TexTransform = MathHelper::Identity4x4();
    skullRitem->ObjCBIndex = 3;
    skullRitem->Mat = mMaterials["skullMat"].get();
    skullRitem->Geo = mGeometries["skullGeo"].get();
    skullRitem->PrimitiveType = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    skullRitem->IndexCount = skullRitem->Geo->DrawArgs["skull"].IndexCount;
    skullRitem->StartIndexLocation = skullRitem->Geo->DrawArgs["skull"].StartIndexLocation;
    skullRitem->BaseVertexLocation = skullRitem->Geo->DrawArgs["skull"].BaseVertexLocation;

	mRitemLayer[(int)RenderLayer::Opaque].push_back(skullRitem.get());
	mAllRitems.push_back(std::move(skullRitem));

    auto gridRitem = std::make_unique<RenderItem>();
    gridRitem->World = MathHelper::Identity4x4();
	XMStoreFloat4x4(&gridRitem->TexTransform, XMMatrixScaling(8.0f, 8.0f, 1.0f));
	gridRitem->ObjCBIndex = 4;
	gridRitem->Mat = mMaterials["tile0"].get();
	gridRitem->Geo = mGeometries["shapeGeo"].get();
	gridRitem->PrimitiveType = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    gridRitem->IndexCount = gridRitem->Geo->DrawArgs["grid"].IndexCount;
    gridRitem->StartIndexLocation = gridRitem->Geo->DrawArgs["grid"].StartIndexLocation;
    gridRitem->BaseVertexLocation = gridRitem->Geo->DrawArgs["grid"].BaseVertexLocation;

	mRitemLayer[(int)RenderLayer::Opaque].push_back(gridRitem.get());
	mAllRitems.push_back(std::move(gridRitem));

	XMMATRIX brickTexTransform = XMMatrixScaling(1.5f, 2.0f, 1.0f);
	UINT objCBIndex = 5;
	for(int i = 0; i < 5; ++i)
	{
		auto leftCylRitem = std::make_unique<RenderItem>();
		auto rightCylRitem = std::make_unique<RenderItem>();
		auto leftSphereRitem = std::make_unique<RenderItem>();
		auto rightSphereRitem = std::make_unique<RenderItem>();

		XMMATRIX leftCylWorld = XMMatrixTranslation(-5.0f, 1.5f, -10.0f + i*5.0f);
		XMMATRIX rightCylWorld = XMMatrixTranslation(+5.0f, 1.5f, -10.0f + i*5.0f);

		XMMATRIX leftSphereWorld = XMMatrixTranslation(-5.0f, 3.5f, -10.0f + i*5.0f);
		XMMATRIX rightSphereWorld = XMMatrixTranslation(+5.0f, 3.5f, -10.0f + i*5.0f);

		XMStoreFloat4x4(&leftCylRitem->World, rightCylWorld);
		XMStoreFloat4x4(&leftCylRitem->TexTransform, brickTexTransform);
		leftCylRitem->ObjCBIndex = objCBIndex++;
		leftCylRitem->Mat = mMaterials["bricks0"].get();
		leftCylRitem->Geo = mGeometries["shapeGeo"].get();
		leftCylRitem->PrimitiveType = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
		leftCylRitem->IndexCount = leftCylRitem->Geo->DrawArgs["cylinder"].IndexCount;
		leftCylRitem->StartIndexLocation = leftCylRitem->Geo->DrawArgs["cylinder"].StartIndexLocation;
		leftCylRitem->BaseVertexLocation = leftCylRitem->Geo->DrawArgs["cylinder"].BaseVertexLocation;

		XMStoreFloat4x4(&rightCylRitem->World, leftCylWorld);
		XMStoreFloat4x4(&rightCylRitem->TexTransform, brickTexTransform);
		rightCylRitem->ObjCBIndex = objCBIndex++;
		rightCylRitem->Mat = mMaterials["bricks0"].get();
		rightCylRitem->Geo = mGeometries["shapeGeo"].get();
		rightCylRitem->PrimitiveType = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
		rightCylRitem->IndexCount = rightCylRitem->Geo->DrawArgs["cylinder"].IndexCount;
		rightCylRitem->StartIndexLocation = rightCylRitem->Geo->DrawArgs["cylinder"].StartIndexLocation;
		rightCylRitem->BaseVertexLocation = rightCylRitem->Geo->DrawArgs["cylinder"].BaseVertexLocation;

		XMStoreFloat4x4(&leftSphereRitem->World, leftSphereWorld);
		leftSphereRitem->TexTransform = MathHelper::Identity4x4();
		leftSphereRitem->ObjCBIndex = objCBIndex++;
		leftSphereRitem->Mat = mMaterials["mirror0"].get();
		leftSphereRitem->Geo = mGeometries["shapeGeo"].get();
		leftSphereRitem->PrimitiveType = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
		leftSphereRitem->IndexCount = leftSphereRitem->Geo->DrawArgs["sphere"].IndexCount;
		leftSphereRitem->StartIndexLocation = leftSphereRitem->Geo->DrawArgs["sphere"].StartIndexLocation;
		leftSphereRitem->BaseVertexLocation = leftSphereRitem->Geo->DrawArgs["sphere"].BaseVertexLocation;

		XMStoreFloat4x4(&rightSphereRitem->World, rightSphereWorld);
		rightSphereRitem->TexTransform = MathHelper::Identity4x4();
		rightSphereRitem->ObjCBIndex = objCBIndex++;
		rightSphereRitem->Mat = mMaterials["mirror0"].get();
		rightSphereRitem->Geo = mGeometries["shapeGeo"].get();
		rightSphereRitem->PrimitiveType = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
		rightSphereRitem->IndexCount = rightSphereRitem->Geo->DrawArgs["sphere"].IndexCount;
		rightSphereRitem->StartIndexLocation = rightSphereRitem->Geo->DrawArgs["sphere"].StartIndexLocation;
		rightSphereRitem->BaseVertexLocation = rightSphereRitem->Geo->DrawArgs["sphere"].BaseVertexLocation;

		mRitemLayer[(int)RenderLayer::Opaque].push_back(leftCylRitem.get());
		mRitemLayer[(int)RenderLayer::Opaque].push_back(rightCylRitem.get());
		mRitemLayer[(int)RenderLayer::Opaque].push_back(leftSphereRitem.get());
		mRitemLayer[(int)RenderLayer::Opaque].push_back(rightSphereRitem.get());

		mAllRitems.push_back(std::move(leftCylRitem));
		mAllRitems.push_back(std::move(rightCylRitem));
		mAllRitems.push_back(std::move(leftSphereRitem));
		mAllRitems.push_back(std::move(rightSphereRitem));
	}
}

void SsaoApp::DrawRenderItems(ID3D12GraphicsCommandList* cmdList, const std::vector<RenderItem*>& ritems)
{
    UINT objCBByteSize = d3dUtil::CalcConstantBufferByteSize(sizeof(ObjectConstants));
 
	auto objectCB = mCurrFrameResource->ObjectCB->Resource();

    // For each render item...
    for(size_t i = 0; i < ritems.size(); ++i)
    {
        auto ri = ritems[i];

        cmdList->IASetVertexBuffers(0, 1, &ri->Geo->VertexBufferView());
        cmdList->IASetIndexBuffer(&ri->Geo->IndexBufferView());
        cmdList->IASetPrimitiveTopology(ri->PrimitiveType);

        D3D12_GPU_VIRTUAL_ADDRESS objCBAddress = objectCB->GetGPUVirtualAddress() + ri->ObjCBIndex*objCBByteSize;

		cmdList->SetGraphicsRootConstantBufferView(0, objCBAddress);

        cmdList->DrawIndexedInstanced(ri->IndexCount, 1, ri->StartIndexLocation, ri->BaseVertexLocation, 0);
    }
}

void SsaoApp::DrawSceneToShadowMap()
{
    mCommandList->RSSetViewports(1, &mShadowMap->Viewport());
    mCommandList->RSSetScissorRects(1, &mShadowMap->ScissorRect());

    // Change to DEPTH_WRITE.
    mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(mShadowMap->Resource(),
        D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATE_DEPTH_WRITE));

    // Clear the back buffer and depth buffer.
    mCommandList->ClearDepthStencilView(mShadowMap->Dsv(), 
        D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 1.0f, 0, 0, nullptr);

    // Specify the buffers we are going to render to.
    mCommandList->OMSetRenderTargets(0, nullptr, false, &mShadowMap->Dsv());

    // Bind the pass constant buffer for the shadow map pass.
    UINT passCBByteSize = d3dUtil::CalcConstantBufferByteSize(sizeof(PassConstants));
    auto passCB = mCurrFrameResource->PassCB->Resource();
    D3D12_GPU_VIRTUAL_ADDRESS passCBAddress = passCB->GetGPUVirtualAddress() + 1*passCBByteSize;
    mCommandList->SetGraphicsRootConstantBufferView(1, passCBAddress);

    mCommandList->SetPipelineState(mPSOs["shadow_opaque"].Get());

    DrawRenderItems(mCommandList.Get(), mRitemLayer[(int)RenderLayer::Opaque]);

    // Change back to GENERIC_READ so we can read the texture in a shader.
    mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(mShadowMap->Resource(),
        D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_GENERIC_READ));
}
 
void SsaoApp::DrawNormalsAndDepth()
{
	mCommandList->RSSetViewports(1, &mScreenViewport);
    mCommandList->RSSetScissorRects(1, &mScissorRect);

	auto normalMap = mSsao->NormalMap();
	auto normalMapRtv = mSsao->NormalMapRtv();
	
    // Change to RENDER_TARGET.
    mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(normalMap,
        D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATE_RENDER_TARGET));

	// Clear the screen normal map and depth buffer.
	float clearValue[] = {0.0f, 0.0f, 1.0f, 0.0f};
    mCommandList->ClearRenderTargetView(normalMapRtv, clearValue, 0, nullptr);
    mCommandList->ClearDepthStencilView(DepthStencilView(), D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 1.0f, 0, 0, nullptr);

	// Specify the buffers we are going to render to.
    mCommandList->OMSetRenderTargets(1, &normalMapRtv, true, &DepthStencilView());

    // Bind the constant buffer for this pass.
    auto passCB = mCurrFrameResource->PassCB->Resource();
    mCommandList->SetGraphicsRootConstantBufferView(1, passCB->GetGPUVirtualAddress());

    mCommandList->SetPipelineState(mPSOs["drawNormals"].Get());

    DrawRenderItems(mCommandList.Get(), mRitemLayer[(int)RenderLayer::Opaque]);

    // Change back to GENERIC_READ so we can read the texture in a shader.
    mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(normalMap,
        D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_GENERIC_READ));
}

CD3DX12_CPU_DESCRIPTOR_HANDLE SsaoApp::GetCpuSrv(int index)const
{
    auto srv = CD3DX12_CPU_DESCRIPTOR_HANDLE(mSrvDescriptorHeap->GetCPUDescriptorHandleForHeapStart());
    srv.Offset(index, mCbvSrvUavDescriptorSize);
    return srv;
}

CD3DX12_GPU_DESCRIPTOR_HANDLE SsaoApp::GetGpuSrv(int index)const
{
    auto srv = CD3DX12_GPU_DESCRIPTOR_HANDLE(mSrvDescriptorHeap->GetGPUDescriptorHandleForHeapStart());
    srv.Offset(index, mCbvSrvUavDescriptorSize);
    return srv;
}

CD3DX12_CPU_DESCRIPTOR_HANDLE SsaoApp::GetDsv(int index)const
{
    auto dsv = CD3DX12_CPU_DESCRIPTOR_HANDLE(mDsvHeap->GetCPUDescriptorHandleForHeapStart());
    dsv.Offset(index, mDsvDescriptorSize);
    return dsv;
}

CD3DX12_CPU_DESCRIPTOR_HANDLE SsaoApp::GetRtv(int index)const
{
    auto rtv = CD3DX12_CPU_DESCRIPTOR_HANDLE(mRtvHeap->GetCPUDescriptorHandleForHeapStart());
    rtv.Offset(index, mRtvDescriptorSize);
    return rtv;
}

std::array<const CD3DX12_STATIC_SAMPLER_DESC, 7> SsaoApp::GetStaticSamplers()
{
	// Applications usually only need a handful of samplers.  So just define them all up front
	// and keep them available as part of the root signature.  

	const CD3DX12_STATIC_SAMPLER_DESC pointWrap(
		0, // shaderRegister
		D3D12_FILTER_MIN_MAG_MIP_POINT, // filter
		D3D12_TEXTURE_ADDRESS_MODE_WRAP,  // addressU
		D3D12_TEXTURE_ADDRESS_MODE_WRAP,  // addressV
		D3D12_TEXTURE_ADDRESS_MODE_WRAP); // addressW

	const CD3DX12_STATIC_SAMPLER_DESC pointClamp(
		1, // shaderRegister
		D3D12_FILTER_MIN_MAG_MIP_POINT, // filter
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,  // addressU
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,  // addressV
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP); // addressW

	const CD3DX12_STATIC_SAMPLER_DESC linearWrap(
		2, // shaderRegister
		D3D12_FILTER_MIN_MAG_MIP_LINEAR, // filter
		D3D12_TEXTURE_ADDRESS_MODE_WRAP,  // addressU
		D3D12_TEXTURE_ADDRESS_MODE_WRAP,  // addressV
		D3D12_TEXTURE_ADDRESS_MODE_WRAP); // addressW

	const CD3DX12_STATIC_SAMPLER_DESC linearClamp(
		3, // shaderRegister
		D3D12_FILTER_MIN_MAG_MIP_LINEAR, // filter
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,  // addressU
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,  // addressV
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP); // addressW

	const CD3DX12_STATIC_SAMPLER_DESC anisotropicWrap(
		4, // shaderRegister
		D3D12_FILTER_ANISOTROPIC, // filter
		D3D12_TEXTURE_ADDRESS_MODE_WRAP,  // addressU
		D3D12_TEXTURE_ADDRESS_MODE_WRAP,  // addressV
		D3D12_TEXTURE_ADDRESS_MODE_WRAP,  // addressW
		0.0f,                             // mipLODBias
		8);                               // maxAnisotropy

	const CD3DX12_STATIC_SAMPLER_DESC anisotropicClamp(
		5, // shaderRegister
		D3D12_FILTER_ANISOTROPIC, // filter
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,  // addressU
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,  // addressV
		D3D12_TEXTURE_ADDRESS_MODE_CLAMP,  // addressW
		0.0f,                              // mipLODBias
		8);                                // maxAnisotropy

    const CD3DX12_STATIC_SAMPLER_DESC shadow(
        6, // shaderRegister
        D3D12_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT, // filter
        D3D12_TEXTURE_ADDRESS_MODE_BORDER,  // addressU
        D3D12_TEXTURE_ADDRESS_MODE_BORDER,  // addressV
        D3D12_TEXTURE_ADDRESS_MODE_BORDER,  // addressW
        0.0f,                               // mipLODBias
        16,                                 // maxAnisotropy
        D3D12_COMPARISON_FUNC_LESS_EQUAL,
        D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK);

	return { 
		pointWrap, pointClamp,
		linearWrap, linearClamp, 
		anisotropicWrap, anisotropicClamp,
        shadow 
    };
}

