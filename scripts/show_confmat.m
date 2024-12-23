function show_confmat(cm_raw, labels, accuracy)
    figure(1)
    % confusionchart(cm_raw, labels, 'normalization', 'row-normalized')
    % title(['Accuracy: ' num2str(accuracy)])
    % 
    % figure(2)
    confusionchart(cm_raw, labels)
    title(['Accuracy: ' num2str(accuracy)])

    set(gca,'FontSize',14);
end