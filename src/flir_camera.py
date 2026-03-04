"""
FLIR Blackfly S camera wrapper using PySpin.
Mimics the cv2.VideoCapture interface so it can be used as a drop-in replacement.
"""

import PySpin
import numpy as np
import cv2
from src.logger import setup_logger

logger = setup_logger("flir_camera")


class FLIRCamera:
    """
    Drop-in replacement for cv2.VideoCapture using the FLIR Blackfly S
    via the Spinnaker PySpin SDK.

    Usage mirrors cv2.VideoCapture:
        cap = FLIRCamera()
        cap.isOpened()     -> bool
        ret, frame = cap.read()   -> (bool, numpy BGR image or None)
        cap.release()
    """

    def __init__(
        self,
        camera_index: int = 0,
        frame_rate: float = 10.0,
        exposure_us: float = None,   # microseconds; None = auto
        gain_db: float = None,       # dB; None = auto
        gamma: float = None,
        width: int = None,           # None = max sensor width
        height: int = None,          # None = max sensor height
        use_polarization_color: bool = True
    ):
        self._system = PySpin.System.GetInstance()
        self._cam_list = self._system.GetCameras()

        if self._cam_list.GetSize() == 0:
            self._cam_list.Clear()
            self._system.ReleaseInstance()
            raise RuntimeError("No FLIR cameras detected.")

        if camera_index >= self._cam_list.GetSize():
            raise RuntimeError(
                f"Camera index {camera_index} out of range "
                f"({self._cam_list.GetSize()} camera(s) found)."
            )

        self._opened = True
        self._cam = self._cam_list.GetByIndex(camera_index)
        self._cam.Init()

        nodemap = self._cam.GetNodeMap()

        # ── Acquisition mode: Continuous ────────────────────────────────
        acq_mode = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))
        continuous = acq_mode.GetEntryByName("Continuous")
        acq_mode.SetIntValue(continuous.GetValue())

        # ── Frame rate ───────────────────────────────────────────────────
        # Enable frame rate control (disable auto frame rate)
        try:
            fra_enable = PySpin.CBooleanPtr(
                nodemap.GetNode("AcquisitionFrameRateEnable")
            )
            if PySpin.IsAvailable(fra_enable) and PySpin.IsWritable(fra_enable):
                fra_enable.SetValue(True)

            fra_node = PySpin.CFloatPtr(nodemap.GetNode("AcquisitionFrameRate"))
            if PySpin.IsAvailable(fra_node) and PySpin.IsWritable(fra_node):
                fra_node.SetValue(frame_rate)
                logger.info(f"Frame rate set to {frame_rate} fps")
        except PySpin.SpinnakerException as e:
            logger.warning(f"Could not set frame rate: {e}")

        # ── Exposure ─────────────────────────────────────────────────────
        try:
            exp_auto = PySpin.CEnumerationPtr(nodemap.GetNode("ExposureAuto"))
            if PySpin.IsAvailable(exp_auto) and PySpin.IsWritable(exp_auto):
                if exposure_us is None:
                    exp_auto.SetIntValue(exp_auto.GetEntryByName("Continuous").GetValue())
                    logger.info("Exposure: Auto (Continuous)")
                else:
                    exp_auto.SetIntValue(exp_auto.GetEntryByName("Off").GetValue())
                    exp_time = PySpin.CFloatPtr(nodemap.GetNode("ExposureTime"))
                    exp_us = max(exp_time.GetMin(), min(exp_time.GetMax(), exposure_us))
                    exp_time.SetValue(exp_us)
                    logger.info(f"Exposure set to {exp_us:.1f} µs")
            else:
                logger.info("ExposureAuto not writable — leaving at camera default")
        except PySpin.SpinnakerException as e:
            logger.warning(f"Could not set exposure: {e}")

        # ── Gain ─────────────────────────────────────────────────────────
        try:
            gain_auto = PySpin.CEnumerationPtr(nodemap.GetNode("GainAuto"))
            if PySpin.IsAvailable(gain_auto) and PySpin.IsWritable(gain_auto):
                if gain_db is None:
                    gain_auto.SetIntValue(gain_auto.GetEntryByName("Continuous").GetValue())
                    logger.info("Gain: Auto (Continuous)")
                else:
                    gain_auto.SetIntValue(gain_auto.GetEntryByName("Off").GetValue())
                    gain_node = PySpin.CFloatPtr(nodemap.GetNode("Gain"))
                    g = max(gain_node.GetMin(), min(gain_node.GetMax(), gain_db))
                    gain_node.SetValue(g)
                    logger.info(f"Gain set to {g:.2f} dB")
            else:
                logger.info("GainAuto not writable — leaving at camera default")
        except PySpin.SpinnakerException as e:
            logger.warning(f"Could not set gain: {e}")
        # ── Gamma ────────────────────────────────────────────────────────────
        try:
            gamma_enable = PySpin.CBooleanPtr(nodemap.GetNode("GammaEnable"))
            if gamma is None:
                # Disable gamma correction (linear response)
                if PySpin.IsAvailable(gamma_enable) and PySpin.IsWritable(gamma_enable):
                    gamma_enable.SetValue(False)
                    logger.info("Gamma: Disabled (linear)")
            else:
                # Enable and set gamma value
                if PySpin.IsAvailable(gamma_enable) and PySpin.IsWritable(gamma_enable):
                    gamma_enable.SetValue(True)
                gamma_node = PySpin.CFloatPtr(nodemap.GetNode("Gamma"))
                if PySpin.IsAvailable(gamma_node) and PySpin.IsWritable(gamma_node):
                    g = max(gamma_node.GetMin(), min(gamma_node.GetMax(), gamma))
                    gamma_node.SetValue(g)
                    logger.info(f"Gamma set to {g:.2f}")
                else:
                    logger.warning("Gamma node not writable")
        except PySpin.SpinnakerException as e:
            logger.warning(f"Could not set gamma: {e}")

        # ── ROI / Image size ─────────────────────────────────────────────
        # (Optional; leave at default for full sensor)
        if width is not None:
            try:
                w_node = PySpin.CIntegerPtr(nodemap.GetNode("Width"))
                w_node.SetValue(
                    max(w_node.GetMin(), min(w_node.GetMax(), width))
                )
            except PySpin.SpinnakerException as e:
                logger.warning(f"Could not set width: {e}")

        if height is not None:
            try:
                h_node = PySpin.CIntegerPtr(nodemap.GetNode("Height"))
                h_node.SetValue(
                    max(h_node.GetMin(), min(h_node.GetMax(), height))
                )
            except PySpin.SpinnakerException as e:
                logger.warning(f"Could not set height: {e}")

        # ── Pixel format: BayerRGPolarized8 for colour-corrected output ──────
        try:
            px_fmt = PySpin.CEnumerationPtr(nodemap.GetNode("PixelFormat"))
            pol8 = px_fmt.GetEntryByName("BayerRGPolarized8")
            if use_polarization_color and PySpin.IsAvailable(pol8) and PySpin.IsReadable(pol8):
                px_fmt.SetIntValue(pol8.GetValue())
                self._pixel_format = "BayerRGPolarized8"
                logger.info("Pixel format: BayerRGPolarized8 (polarization colour correction enabled)")
            else:
                # Fallback to plain BayerRG8
                bayer = px_fmt.GetEntryByName("BayerRG8")
                px_fmt.SetIntValue(bayer.GetValue())
                self._pixel_format = "BayerRG8"
                logger.warning("BayerRGPolarized8 not available, falling back to BayerRG8")
        except PySpin.SpinnakerException as e:
            self._pixel_format = "unknown"
            logger.warning(f"Could not set pixel format: {e}")

        # ── Start acquisition ────────────────────────────────────────────
        self._cam.BeginAcquisition()
        logger.info("FLIR camera acquisition started.")

    # ── Public interface (mirrors cv2.VideoCapture) ──────────────────────

    def isOpened(self) -> bool:
        return self._opened

    def read(self):
        """
        Returns (True, BGR numpy array) on success, (False, None) on failure.
        Mirrors cv2.VideoCapture.read().
        """
        if not self._opened:
            return False, None
        try:
            image_result = self._cam.GetNextImage(5000)  # 5 s timeout

            if image_result.IsIncomplete():
                logger.warning(
                    f"Image incomplete: {image_result.GetImageStatus()}"
                )
                image_result.Release()
                return False, None

            if self._pixel_format == "BayerRGPolarized8":
                from src.polarization_color import extract_color
                h = image_result.GetHeight()
                w = image_result.GetWidth()
                raw = np.frombuffer(image_result.GetData(), dtype=np.uint8).reshape(h, w)
                result = extract_color(raw)
                frame = result['color']
                
            elif self._pixel_format == "BayerRG8":
                mono = image_result.GetNDArray()
                frame = cv2.cvtColor(mono, cv2.COLOR_BayerRG2BGR)
            else:
                converted = image_result.Convert(PySpin.PixelFormat_BGR8, PySpin.HQ_LINEAR)
                frame = converted.GetNDArray()

            image_result.Release()
            return True, frame

        except PySpin.SpinnakerException as e:
            logger.error(f"Failed to grab frame: {e}")
            return False, None

    def release(self):
        """Stop acquisition and clean up."""
        if self._opened:
            try:
                self._cam.EndAcquisition()
            except Exception:
                pass
            try:
                self._cam.DeInit()
            except Exception:
                pass
            self._opened = False

        if hasattr(self, '_cam'):
            del self._cam
        if hasattr(self, '_cam_list'):
            self._cam_list.Clear()
        if hasattr(self, '_system'):
            self._system.ReleaseInstance()
        logger.info("FLIR camera released.")

    def __del__(self):
        self.release()
