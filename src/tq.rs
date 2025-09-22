use std::fs;
use std::path::Path;

pub fn run(
    chunks: &[crate::chunk::Chunk],
    inf: &crate::ffms::VidInf,
    args: &crate::Args,
    tq_range: (f64, f64),
    qp_range: (f64, f64),
    work_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let idx = crate::ffms::VidIdx::new(&args.input, args.quiet)?;
    let (tq_min, tq_max) = tq_range;
    let (qp_min, qp_max) = qp_range;
    let mut crfs = vec![];
    let mut scores = vec![];
    let target = f64::midpoint(tq_min, tq_max);

    for round in 0..6 {
        let crf: f64 = if round == 0 {
            f64::midpoint(qp_min, qp_max)
        } else if round == 1 {
            if scores[0] < target {
                f64::max(crfs[0] - (qp_max - qp_min) / 4.0, qp_min)
            } else {
                f64::min(crfs[0] + (qp_max - qp_min) / 4.0, qp_max)
            }
        } else {
            interpolate(round - 1, &crfs, &scores, target)?.clamp(qp_min, qp_max)
        };

        let crf_rounded = (crf * 4.0).round() / 4.0;
        println!("\rTQ Round {}: Testing CRF {:.2}", round + 1, crf_rounded);

        crate::svt::encode_all(chunks, inf, args, crf_rounded as f32, &idx, work_dir);
        crate::chunk::merge_out(&work_dir.join("split"), &work_dir.join("test.mkv"), inf)?;

        let score = crate::vship::test_quality(
            &args.input,
            &work_dir.join("test.mkv"),
            round == 0,
            work_dir,
        )?;

        crfs.push(crf_rounded);
        scores.push(score);

        if score >= tq_min && score <= tq_max {
            copy_to_encode(work_dir)?;
            return Ok(());
        }

        if crf_rounded <= qp_min || crf_rounded >= qp_max {
            copy_to_encode(work_dir)?;
            return Ok(());
        }
    }

    copy_to_encode(work_dir)?;
    Ok(())
}

fn interpolate(
    method: usize,
    crfs: &[f64],
    scores: &[f64],
    target: f64,
) -> Result<f64, Box<dyn std::error::Error>> {
    let mut sorted: Vec<_> = crfs.iter().zip(scores).map(|(c, s)| (*s, *c)).collect();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let (s, c): (Vec<f64>, Vec<f64>) = sorted.into_iter().unzip();

    Ok(match method {
        1 => crate::interp::lerp(&[s[0], s[1]], &[c[0], c[1]], target),
        2 => crate::interp::natural_cubic(&s, &c, target),
        3 => {
            let s4: [f64; 4] = s[..4].try_into()?;
            let c4: [f64; 4] = c[..4].try_into()?;
            crate::interp::pchip(&s4, &c4, target)
        }
        4 => {
            let s5: [f64; 5] = s[..5].try_into()?;
            let c5: [f64; 5] = c[..5].try_into()?;
            crate::interp::akima(&s5, &c5, target)
        }
        _ => None,
    }
    .unwrap_or(crfs[crfs.len() - 1]))
}

fn copy_to_encode(work_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    for entry in fs::read_dir(work_dir.join("split"))? {
        let entry = entry?;
        if entry.path().extension().is_some_and(|ext| ext == "ivf") {
            fs::copy(entry.path(), work_dir.join("encode").join(entry.file_name()))?;
        }
    }
    Ok(())
}
