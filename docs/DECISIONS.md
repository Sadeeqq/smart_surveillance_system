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

## Open decisions to validate

| Question | Why unresolved | Planned resolution |
|---|---|---|
| Is the project to be distributed under AGPL-3.0 if Ultralytics YOLO is used, or is a different model/licence path required? | Ultralytics says its code and trained models are AGPL-3.0 by default and this can impose source-release obligations on the combined application. | Review before distributing a build that includes or uses Ultralytics. |
| Which compatible ByteTrack implementation? | Must validate package compatibility with detector/runtime. | Phase 3. |
| Which dedicated face detector and whether DeepFace + FaceNet is viable? | Explicit candidate; licensing, dependency, runtime and false-match checks required. | Phase 5 evaluation on authorized samples. |
| Which camera/OS combinations will be listed as supported? | Compatibility varies by DroidCam client and transport. | Test and document actual combinations in Phase 1 and later platform checks. |
| What retention duration and operator access policy apply? | PRD leaves retention and roles open and requires definition before real deployment. | Use conservative prototype handling; decide before any real deployment. |
| What debounce, re-arm and recognition thresholds are suitable? | Depend on representative footage and model behaviour; must not be guessed as accuracy guarantees. | Define and measure in Phases 4 and 5. |

The Phase 2 candidate is documented above; it is not represented as production-ready or legally approved for every deployment. No untested platform is presented as supported.
