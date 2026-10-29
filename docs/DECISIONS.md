# Architecture Decisions and Open Questions

## Recorded decisions

| Decision | Rationale / source |
|---|---|
| Use Python with OpenCV, Streamlit, SQLite and pytest in the prototype | These are the PRD's settled stack choices. |
| Keep camera sources behind an adapter | Required for DroidCam initial support and replaceable sources. |
| Use a normalized virtual line and bottom-centre person anchor | Roadmap's starting convention; line is not a physical distance. |
| Crossing is the event trigger; face recognition and Telegram are optional | PRD's operating principles and failure-isolation requirements. |
| Keep images as local files and metadata in SQLite by default | PRD data model and persistence direction. |
| Python 3.12 is the initial development baseline | Available in the project environment; dependency compatibility will be validated in relevant phases. |

## Open decisions to validate

| Question | Why unresolved | Planned resolution |
|---|---|---|
| Which Ultralytics YOLO model/weights and license terms? | PRD calls this a candidate requiring license/runtime validation. | Phase 2 review before selecting weights. |
| Which compatible ByteTrack implementation? | Must validate package compatibility with detector/runtime. | Phase 3. |
| Which dedicated face detector and whether DeepFace + FaceNet is viable? | Explicit candidate; licensing, dependency, runtime and false-match checks required. | Phase 5 evaluation on authorized samples. |
| Which camera/OS combinations will be listed as supported? | Compatibility varies by DroidCam client and transport. | Test and document actual combinations in Phase 1 and later platform checks. |
| What retention duration and operator access policy apply? | PRD leaves retention and roles open and requires definition before real deployment. | Use conservative prototype handling; decide before any real deployment. |
| What debounce, re-arm and recognition thresholds are suitable? | Depend on representative footage and model behaviour; must not be guessed as accuracy guarantees. | Define and measure in Phases 4 and 5. |

No candidate model or untested platform is presented as selected or supported in this foundation phase.
