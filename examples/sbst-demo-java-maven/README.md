# SBST Demo Java SUT (Maven + JaCoCo)

This is a tiny Java System-Under-Test (SUT) included in-repo to provide a reproducible end-to-end target for the SBST TDG pipeline.

## Build + coverage (manual)

From this folder:

- `mvn test jacoco:report`
- JaCoCo XML is written to: `target/site/jacoco/jacoco.xml`

## Designed for the SBST generator

The current MVP generator calls **public no-arg constructors** and **public zero-arg methods** via reflection.
So the production classes expose small, safe, zero-arg methods that exercise `if/else` and `switch` branches.
