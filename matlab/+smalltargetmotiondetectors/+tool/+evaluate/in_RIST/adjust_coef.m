function hModel = adjust_coef(hModel, modelName, speedCoef)
    switch modelName
        case 'ESTMD'
            propertiesList = {
                'hModel.hLamina.hGammaBankPassFilter.hGammaDelay1.order', ...
                'hModel.hLamina.hGammaBankPassFilter.hGammaDelay1.tau', ...
                'hModel.hLamina.hGammaBankPassFilter.hGammaDelay2.order', ...
                'hModel.hLamina.hGammaBankPassFilter.hGammaDelay2.tau', ...
                'hModel.hMedulla.hTm1.hGammaDelay.order', ...
                'hModel.hMedulla.hTm1.hGammaDelay.tau' ...
                };
        case {'DSTMD', 'STMDPlus'}
            propertiesList = {
                'hModel.hLamina.hGammaBankPassFilter.hGammaDelay1.order', ...
                'hModel.hLamina.hGammaBankPassFilter.hGammaDelay1.tau', ...
                'hModel.hLamina.hGammaBankPassFilter.hGammaDelay2.order', ...
                'hModel.hLamina.hGammaBankPassFilter.hGammaDelay2.tau', ...
                'hModel.hMedulla.hDelay4Mi1.hGammaDelay.order', ...
                'hModel.hMedulla.hDelay4Mi1.hGammaDelay.tau', ...
                'hModel.hMedulla.hDelay5Tm1.hGammaDelay.order', ...
                'hModel.hMedulla.hDelay5Tm1.hGammaDelay.order', ...
                'hModel.hMedulla.hDelay6Tm1.hGammaDelay.order', ...
                'hModel.hMedulla.hDelay6Tm1.hGammaDelay.order'
                };
        case 'ApgSTMD'
            propertiesList = {
                'hModel.hLamina.hGammaBankPassFilter.hGammaDelay1.order', ...
                'hModel.hLamina.hGammaBankPassFilter.hGammaDelay1.tau', ...
                'hModel.hLamina.hGammaBankPassFilter.hGammaDelay2.order', ...
                'hModel.hLamina.hGammaBankPassFilter.hGammaDelay2.tau', ...
                'hModel.hMedulla.hDelay4Mi1.hGammaDelay.order', ...
                'hModel.hMedulla.hDelay4Mi1.hGammaDelay.tau', ...
                'hModel.hMedulla.hDelay5Tm1.hGammaDelay.order', ...
                'hModel.hMedulla.hDelay5Tm1.hGammaDelay.order', ...
                'hModel.hMedulla.hDelay6Tm1.hGammaDelay.order', ...
                'hModel.hMedulla.hDelay6Tm1.hGammaDelay.order', ...
                'hModel.hPredictionPathway.deltaT'...
                };
        case 'FracSTMD'
            propertiesList = {
                'hModel.hMedulla.hTm1.hGammaDelay.order', ...
                'hModel.hMedulla.hTm1.hGammaDelay.tau'...
                };
        case 'FeedbackSTMD'
            propertiesList = {
                'hModel.hLamina.hGammaBankPassFilter.hGammaDelay1.order', ...
                'hModel.hLamina.hGammaBankPassFilter.hGammaDelay1.tau', ...
                'hModel.hLamina.hGammaBankPassFilter.hGammaDelay2.order', ...
                'hModel.hLamina.hGammaBankPassFilter.hGammaDelay2.tau', ...
                'hModel.hMedulla.hTm1.hGammaDelay.order', ...
                'hModel.hMedulla.hTm1.hGammaDelay.tau', ...
                'hModel.hLobula.hGammaDelay.order', ...
                'hModel.hLobula.hGammaDelay.tau' ...
                };
        case 'FSTMD'
            propertiesList = {
                'hModel.hLamina.hGammaBankPassFilter.hGammaDelay1.order', ...
                'hModel.hLamina.hGammaBankPassFilter.hGammaDelay1.tau', ...
                'hModel.hLamina.hGammaBankPassFilter.hGammaDelay2.order', ...
                'hModel.hLamina.hGammaBankPassFilter.hGammaDelay2.tau', ...
                'hModel.hMedulla.hTm1.hGammaDelay.order', ...
                'hModel.hMedulla.hTm1.hGammaDelay.tau', ...
                'hModel.hFeedbackPathway.hGammaDelay.order', ...
                'hModel.hFeedbackPathway.hGammaDelay.tau' ...
                };
            
        otherwise
            return;
    end

    
    for nProperty = propertiesList
        property = nProperty{1};
        eval([property, ' = round(', property, ' * speedCoef);']);
        if eval(property) <= 0
            eval([property, ' = 1;']);
        end
        if startsWith(property, 'hModel.hLamina.hGammaBankPassFilter.hGammaDelay2')
            if eval([property, '<= 1'])
                eval([property, ' = 2;']);
            end
        end
    end
         
end