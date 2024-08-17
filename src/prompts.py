CURRENT_BEST = """
       Based on the social media post provided in the user prompt, return the category that the post belongs to based on
       the following four categories and their definitions.
    ---
     Indicator:
    Definition: The post or content lacks any explicit expression concerning suicide. There may be general expressions of distress or sadness, but no direct mention or indication of suicidal thoughts or feelings. May include also a vague feeling about dyning or death but no explicit mentioning of suicide. Note that the mentioning of suicide need to be very explicit in order to not be 'indicator'. Very large general dissatisfaction about life and hopefullness is still 'indicator' when suicide/killing oneself is not mentioned. When in doubt whether the suicide was explicitly mentioned or not choose 'indicator' over 'ideation
    Implication: Individuals categorized under this may not be considering suicide, but it does not rule out the possibility of mental distress. Monitoring and further assessment might be necessary if other signs of depression or mental health issues are present.


    Ideation:
    Definition: The content includes explicit expressions of suicidal thoughts or feelings, but without any specific plan to commit suicide. This can range from vague thoughts about not wanting to live to stronger desires to die.
    Implication: Suicidal ideation, even without a plan, is a significant risk factor for suicide and should be taken seriously. Intervention strategies might involve counseling, therapy, and monitoring to prevent progression to more serious risk levels.


    Behavior:
    Definition: The post includes explicit expressions of suicidal thoughts and also mentions a plan to commit suicide or exhibits self-harming behavior. It must include some form of explicit planning like an explcit method or preperations taken (e.g. suicide note). When you are unsure whether there is a explicit plan or not choose 'ideation' over 'behavior'
    Implication: This category implies an immediate risk for suicide. It often warrants urgent mental health interventions such as crisis response services, safety planning, possibly medication, and intensive therapy or hospitalization to ensure safety.


    Attempt:
    Definition: The content describes past attempts at suicide. This category is focused on historic actions rather than current plans. A concrete action need to have happened in the past (e.g. overdose). When someome merely thought of an attempt in the past this classifies not as an 'attempt'
    Implication: Individuals who have attempted suicide in the past are at a higher risk for future attempts. This necessitates a comprehensive treatment approach, which may include long-term psychotherapy, support groups, and close monitoring of mental health.
    Note that when a post refers to past atempts but also mentions current plans than label it as 'attempt'
    ---
    Note that the suicide risk only corresponds to the person writing the post not of other people potentially being mentioned.
       Only answer with one word. It should be always one of the following  'indicator', 'ideation', 'behaviour', 'attempt'   
       Never answer with something different than one of the four options. Never answer with NaN or empty answer
       """

NO_IMPLICATION = """
       Based on the social media post provided in the user prompt, return the category that the post belongs to based on
       the following four categories and their definitions.
    ---
     Indicator:
    Definition: The post or content lacks any explicit expression concerning suicide. There may be general expressions of distress or sadness, but no direct mention or indication of suicidal thoughts or feelings. May include also a vague feeling about dyning or death but no explicit mentioning of suicide. Note that the mentioning of suicide need to be very explicit in order to not be 'indicator'. Very large general dissatisfaction about life and hopefullness is still 'indicator' when suicide/killing oneself is not mentioned. When in doubt whether the suicide was explicitly mentioned or not choose 'indicator' over 'ideation


    Ideation:
    Definition: The content includes explicit expressions of suicidal thoughts or feelings, but without any specific plan to commit suicide. This can range from vague thoughts about not wanting to live to stronger desires to die.

    Behavior:
    Definition: The post includes explicit expressions of suicidal thoughts and also mentions a plan to commit suicide or exhibits self-harming behavior. It must include some form of explicit planning like an explcit method or preperations taken (e.g. suicide note). When you are unsure whether there is a explicit plan or not choose 'ideation' over 'behavior'


    Attempt:
    Definition: The content describes past attempts at suicide. This category is focused on historic actions rather than current plans. A concrete action need to have happened in the past (e.g. overdose). When someome merely thought of an attempt in the past this classifies not as an 'attempt'
    Note that when a post refers to past attempts but also mentions current plans than label it as 'attempt'
    ---
    Note that the suicide risk only corresponds to the person writing the post not of other people potentially being mentioned.
       Only answer with one word. It should be always one of the following  'indicator', 'ideation', 'behaviour', 'attempt'   
       Never answer with something different than one of the four options. Never answer with NaN or empty answer
       """

# current best
NO_IMPLICATION_SPELLING_FIXED = """
       Based on the social media post provided in the user prompt, return the category that the post belongs to based on
       the following four categories and their definitions.
    ---
     Indicator:
    Definition: The post or content lacks any explicit expression concerning suicide. There may be general expressions of distress or sadness, but no direct mention or indication of suicidal thoughts or feelings. May include also a vague feeling about dying or death but no explicit mentioning of suicide. Note that the mentioning of suicide need to be very explicit in order to not be 'indicator'. Very large general dissatisfaction about life and hopefullness is still 'indicator' when suicide/killing oneself is not mentioned. When in doubt whether the suicide was explicitly mentioned or not choose 'indicator' over 'ideation'.


    Ideation:
    Definition: The content includes explicit expressions of suicidal thoughts or feelings, i.e. a desire to not be alive any longer, but without any specific plan to commit suicide. This can range from vague thoughts about not wanting to live to stronger, more explicit desires to end ones life (albeit without a specific plan how to do so). If no desire to die or to commit suicide is expresssed, consider it 'indicator'. If the post contains specific ideas of how to commit suicide (ways/locations/means/methods) consider it 'behavior'.  

    Behavior:
    Definition: The post includes explicit expressions of suicidal thoughts and also mentions (even if only briefly) a plan to commit suicide or exhibits self-harming behavior. It must include some form of explicit planning like an specific method or preperations taken (e.g. suicide note, lethal medication/drugs, tools/weapons (e.g. knifes/guns/ropes) suitable to end ones life, suitable locations (e.g. bridges/cliffs/buildings to jump off from, train lines to get run over by). if you are unsure whether there is a explicit plan or not choose 'ideation' over 'behavior'.


    Attempt:
    Definition: The content describes past attempts at suicide. This category is focused on historic actions rather than current plans. A concrete action need to have happened in the past (e.g. overdose). When someome merely thought of an attempt in the past this classifies not as an 'attempt'
    Note that when a post refers to past attempts but also mentions current plans than label it as 'attempt'
    ---
    Note that the suicide risk only corresponds to the person writing the post not of other people potentially being mentioned.
       Only answer with one word. It should be always one of the following  'indicator', 'ideation', 'behaviour', 'attempt'   
       Never answer with something different than one of the four options. Never answer with NaN or empty answer
       """


NO_IMPLICATION_SPELLING_FIXED_MORE = """
       Based on the social media post provided in the user prompt, return the category that the post belongs to based on
       the following four categories and their definitions.
    ---
     Indicator:
    Definition: The post or content lacks any explicit expression concerning suicide. There may be general expressions of distress or sadness, but no direct mention or indication of suicidal thoughts or feelings. May include also a vague feeling about dying or death but no explicit mentioning of suicide. Note that the mentioning of suicide need to be very explicit in order to not be 'indicator'. Very large general dissatisfaction about life and hopefullness is still 'indicator' when suicide/killing oneself is not mentioned. When in doubt whether the suicide was explicitly mentioned or not choose 'indicator' over 'ideation'.


    Ideation:
    Definition: The content includes explicit expressions of suicidal thoughts or feelings, i.e. a desire to not be alive any longer, but without any specific plan to commit suicide. This can range from vague thoughts about not wanting to live to stronger, more explicit desires to end ones life (albeit without a specific plan how to do so). If no desire to die or to commit suicide is expresssed, consider it 'indicator'. Also statements denying the intention to commit suicide ('I wont commit suicicde/do it') should be considered 'indicator'. If the post contains specific ideas of how to commit suicide (ways/locations/means/methods) consider it 'behavior'.  

    Behavior:
    Definition: The post includes explicit expressions of suicidal thoughts and also mentions (even if only briefly) a plan to commit suicide or exhibits self-harming behavior. It must include some form of explicit planning like an specific method or preperations taken (e.g. suicide note, lethal medication/drugs, tools/weapons (e.g. knifes/guns/ropes) suitable to end ones life, suitable locations (e.g. bridges/cliffs/buildings to jump off from, train lines to get run over by). if you are unsure whether there is a explicit plan or not choose 'ideation' over 'behavior'.


    Attempt:
    Definition: The content describes past attempts at suicide. This category is focused on historic actions rather than current plans. A concrete action need to have happened in the past (e.g. overdose). When someome merely thought of an attempt in the past this classifies not as an 'attempt'
    Note that when a post refers to past attempts but also mentions current plans than label it as 'attempt'
    ---
    Note that the suicide risk only corresponds to the person writing the post not of other people potentially being mentioned.
       Only answer with one word. It should be always one of the following  'indicator', 'ideation', 'behaviour', 'attempt'   
       Never answer with something different than one of the four options. Never answer with NaN or empty answer
       """
