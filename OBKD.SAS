
/* relative paths? */

PROC IMPORT DATAFILE="\pums_data_for_OBKD.csv"
    OUT=pums_data
    DBMS=CSV
    REPLACE;
    GETNAMES=YES;
RUN;

/* reference group */
%LET ref_group = White non-Hispanic Male;

/* Find all groups */
PROC SQL;
    CREATE TABLE groups AS
    SELECT DISTINCT race_ethnicity_sex
    FROM pums_data
    WHERE race_ethnicity_sex NE "&ref_group.";
QUIT;

/*  Loop over groups and run Oaxaca   */

/* Create macro to run decomposition */
%MACRO run_oaxaca(target_group);

    %PUT Running Oaxaca for &target_group.;

    /* Filter data to only ref group and target group */
    DATA analysis_data;
        SET pums_data;
        IF race_ethnicity_sex = "&ref_group." THEN group = 0;
        ELSE IF race_ethnicity_sex = "&target_group." THEN group = 1;
        ELSE DELETE;
    RUN;

    /* Check how many obs */
    PROC SQL;
        SELECT COUNT(*) INTO :nobs FROM analysis_data;
    QUIT;

    %IF &nobs. < 30 %THEN %DO;
        %PUT WARNING: Not enough obs for group &target_group.;
        %RETURN;
    %END;

    /* Run Oaxaca Decomposition */
    PROC CAUSALMED DATA=analysis_data;
        MODEL log_WAGP = AGEP AGE_SQUARED ESR FOD1P INDP NATIVITY NOC OCCP STATE WKHP WKWN;
        TREAT group;
        DECOMP;
        OUTPUT OUT=oaxaca_results_&target_group.;
    RUN;

    /* Save results */
    PROC EXPORT DATA=oaxaca_results_&target_group.
        OUTFILE="\oaxaca_results_&target_group..csv"
        DBMS=CSV
        REPLACE;
    RUN;

%MEND;


/* run the macro for each group */
DATA _NULL_;
    SET groups;
    CALL EXECUTE('%run_oaxaca(' || trim(race_ethnicity_sex) || ')');
RUN;
