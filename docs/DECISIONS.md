# Architecture Decisions and Open Questions

## Recorded decisions

| Decision | Rationale / source |
|---|---|
| Use Python with OpenCV, Streamlit, SQLite and pytest in the prototype | These are the PRD's settled stack choices. |
| Use `cv2.VideoCapture` behind an adapter for device indices, device paths, stream URLs and recorded videos | Keeps source-specific access replaceable and allows recorded-video fallback; the stream transport must be verified for each source. |
| Keep camera sources behind an adapter | Required for DroidCam initial support and replaceable sources. |
| Use a normalized virtual line and bottom-centre person anchor | Roadmap's starting convention; line is not a physical distance. |
| Crossing is the event trigger; face recognition and Telegram are optional | PRD's operating principles and failure-isolation requirements. |
| Keep images as local files and metadata in SQLite by default | PRD data model and persistence direction. |
| Python 3.12 is the initial development baseline | Available in the project environment; dependency compatibility will be validated in relevant phases. |
| Use the small Ultralytics YOLO11n detection checkpoint as the Phase 2 prototype candidate, without committing its weights | It is the smallest YOLO11 detection model listed by Ultralytics and provides a pretrained COCO person class. This is only a prototype candidate; CPU performance and suitability on the target camera footage need measurement. Ultralytics states its code and trained models are AGPL-3.0 by default. Distribution under that route requires licensing the complete application accordingly; private/commercial use may require an Enterprise licence. |
| Keep person detection behind an injectable detector interface and leave model loading optional/lazy | Keeps the camera path usable without a model installation and makes detector unit tests independent of large model packages and weights. |
| Use Ultralytics' supported `model.track()` API with ByteTrack for Phase 3, with persistent state scoped to one video source | Reuses the installed detector runtime, avoids a second tracking dependency, and provides temporary IDs across consecutive frames. Tracker settings can be selected centrally and overridden with the Ultralytics tracker YAML configuration. |
| Use normalized endpoint coordinates and pixel-scaled signed distance for Phase 4 boundary geometry | The boundary retains relative placement across frame resolutions, while distance and the neutral zone are evaluated in the current frame's pixel geometry. A-to-B is negative-to-positive relative to the ordered endpoints. |
| Use a local JSON boundary settings file for Phase 4, keyed by a hash of the camera source | Enables per-camera save/reload before SQLite is introduced in Phase 6, without persisting stream URLs or credentials. The file lives under Git-ignored `data/`. |
| Use OpenCV Zoo YuNet as the Phase 5 face detector candidate behind a small adapter | The OpenCV runtime already exists, the Zoo publishes an MIT-licensed lightweight detector model, and the detector is replaceable. The local ONNX file is not committed. Validate detection and quality thresholds on consented representative footage before deployment. |
| Keep recognition optional and lazy; implement the PRD's DeepFace + FaceNet candidate behind a recognizer contract | Avoids requiring a large additional runtime for camera/boundary work. Candidate verification needs reference enrollment, quality, runtime, false-match and model-weight license evaluation. Recognition may be omitted or replaced without changing crossing events. |
| Start Phase 4 debounce with a 1% frame-diagonal neutral zone, two-frame initial-side and crossing confirmation, five-second per-track cooldown, and 30-frame lost-track retention | Provides explicit, testable prototype defaults. These values require evaluation on representative camera footage and are not accuracy guarantees. |

## Open decisions to validate

| Question | Why unresolved | Planned resolution |
|---|---|---|
| Is the project to be distributed under AGPL-3.0 if Ultralytics YOLO is used, or is a different model/licence path required? | Ultralytics says its code and trained models are AGPL-3.0 by default and this can impose source-release obligations on the combined application. | Review before distributing a build that includes or uses Ultralytics. |
| Is YuNet suitable for the target cameras, and is DeepFace + FaceNet acceptable for the intended distribution? | Detector performance and recognition false matches are not established by API smoke tests; DeepFace's underlying weights have separate license terms and installation/runtime costs. | Measure on consented, representative samples; document model/runtime/license and false-match limits before deployment. |
| Which camera/OS combinations will be listed as supported? | Compatibility varies by DroidCam client and transport. | Test and document actual combinations in Phase 1 and later platform checks. |
| What retention duration and operator access policy apply? | PRD leaves retention and roles open and requires definition before real deployment. | Use conservative prototype handling; decide before any real deployment. |
| What debounce values are suitable for deployment footage? | Current Phase 4 values are prototype defaults and have not been tuned on representative deployment footage. | Measure false/missed crossings on recorded and live footage in Phases 4 and 8. |

The Phase 2 candidate is documented above; it is not represented as production-ready or legally approved for every deployment. No untested platform is presented as supported.
