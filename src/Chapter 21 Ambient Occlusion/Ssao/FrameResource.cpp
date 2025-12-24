#include "FrameResource.h"

FrameResource::FrameResource(ID3D12Device* device, UINT passCount, UINT objectCount, UINT materialCount)
{
    ThrowIfFailed(device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT,
		IID_PPV_ARGS(CmdListAlloc.GetAddressOf())));

    PassCB = std::make_unique<UploadBuffer<PassConstants>>(device, passCount, true);
    SsaoCB = std::make_unique<UploadBuffer<SsaoConstants>>(device, 1, true);
	MaterialBuffer = std::make_unique<UploadBuffer<MaterialData>>(device, materialCount, false);
    ObjectCB = std::make_unique<UploadBuffer<ObjectConstants>>(device, objectCount, true);

    // MB часть
    MotionBlurCB = std::make_unique<UploadBuffer<MB_velocityStage_CB>>(device, 1, true);
    tileMaxCB = std::make_unique<UploadBuffer<MB_tileMaxStage_CB>>(device, 1, true);
    NeighbourMaxCB = std::make_unique<UploadBuffer<MB_neighbourMaxStage_CB>>(device, 1, true);
    MotionBlurFinalCB = std::make_unique<UploadBuffer<MB_motionBlurFinalStage_CB>>(device, 1, true);
}

FrameResource::~FrameResource()
{

}