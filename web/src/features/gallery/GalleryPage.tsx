import { CheckCircleIcon, DashedCircleIcon, ExternalIcon } from "../../components/icons";
import { docUrl } from "../../lib/links";
import { Link } from "../../lib/router";
import { usePageTitle } from "../../lib/usePageTitle";
import { OrbitThumbnail } from "./OrbitThumbnail";
import { STUDIES, STUDY_STATUS_LABEL, type StudySummary } from "./studies";

function StudyStatus({ study }: { study: StudySummary }) {
  const active = study.status === "active";
  const Icon = active ? CheckCircleIcon : DashedCircleIcon;
  return (
    <span className={`inline-flex items-center gap-1.5 text-sm font-medium ${active ? "text-valid" : "text-graphite"}`}>
      <Icon />
      {STUDY_STATUS_LABEL[study.status]}
    </span>
  );
}

function FeaturedStudy({ study }: { study: StudySummary }) {
  return (
    <article aria-labelledby="featured-title" className="grid gap-8 border-y border-rule py-10 lg:grid-cols-12 lg:gap-12">
      <div className="lg:col-span-5">
        <StudyStatus study={study} />
        <h2 id="featured-title" className="mt-3 text-4xl sm:text-5xl">
          {study.title}
        </h2>
        <p className="mt-5 text-lg leading-relaxed">{study.question}</p>
        <p className="mt-4 text-graphite">{study.progress}</p>
        <div className="mt-7 flex flex-wrap items-center gap-x-6 gap-y-3">
          <Link
            to={study.route ?? "/"}
            className="inline-flex items-center rounded-sm bg-ink px-4 py-2.5 font-medium text-sheet no-underline hover:bg-canvas-raised"
          >
            Open the orbit viewer
          </Link>
          <a className="prose-link inline-flex items-center gap-1.5 text-graphite" href={docUrl(study.specPath)} rel="noreferrer">
            Study specification
            <ExternalIcon className="size-3.5" />
          </a>
        </div>
      </div>
      <figure className="lg:col-span-7">
        <Link to={study.route ?? "/"} tabIndex={-1} aria-hidden="true" className="block">
          <OrbitThumbnail />
        </Link>
        <figcaption className="mt-3 text-sm text-graphite">
          The validated reference orbit around L1 in the rotating Earth–Moon frame, with equal x and y scale. It checks the physics
          engine; it is not yet a result of the study.
        </figcaption>
      </figure>
    </article>
  );
}

function PlannedStudy({ study }: { study: StudySummary }) {
  return (
    <li className="grid gap-3 border-b border-rule py-7 md:grid-cols-12 md:gap-8">
      <div className="md:col-span-4">
        <h3 className="text-2xl">{study.title}</h3>
        <div className="mt-2">
          <StudyStatus study={study} />
        </div>
      </div>
      <div className="md:col-span-8">
        <p>{study.question}</p>
        <p className="mt-2 text-graphite">
          {study.progress}{" "}
          <a className="prose-link inline-flex items-center gap-1 whitespace-nowrap" href={docUrl(study.specPath)} rel="noreferrer">
            Read the specification
            <ExternalIcon className="size-3.5" />
          </a>
        </p>
      </div>
    </li>
  );
}

export function GalleryPage() {
  usePageTitle("Phase Space");
  const [featured, ...planned] = STUDIES;
  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-8">
      <section aria-labelledby="mission" className="max-w-4xl pb-14 pt-14 sm:pt-20">
        <h1 id="mission" className="text-[2.6rem] leading-[1.08] sm:text-6xl">
          A small laboratory for questions that can be answered honestly.
        </h1>
        <p className="mt-7 max-w-[62ch] text-lg leading-relaxed text-graphite sm:text-xl">
          Phase Space runs a few bounded computational studies in search, optimization and learning. Each one asks a
          falsifiable question, compares against credible baselines at matched compute, and publishes its evidence as
          versioned data you can inspect here, null results included.
        </p>
      </section>

      {featured !== undefined && <FeaturedStudy study={featured} />}

      <section aria-labelledby="planned" className="mt-16">
        <h2 id="planned" className="text-3xl">
          Planned studies
        </h2>
        <p className="mt-3 max-w-[62ch] text-graphite">
          These have written specifications but no data yet. They will open here once there is something real to inspect.
        </p>
        <ul className="mt-6 border-t border-rule">
          {planned.map((study) => (
            <PlannedStudy key={study.study_id} study={study} />
          ))}
        </ul>
      </section>
    </div>
  );
}
